from langgraph.graph import StateGraph, START, END
from app.backend.state import GaurdrailState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from typing import Optional, List

from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY
from pydantic import BaseModel

from app.backend.db import save_message
from app.backend.langgraph_flow import get_graph

from langgraph.types import Command

graph = get_graph()

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

class ValidatorSchema(BaseModel):
    allowed: bool
    reason: str

def build_prompt(user_input: str, images_list: Optional[List[str]] = None, thread_id: str = None) -> str:
    """
    Build a structured prompt so downstream tools can reliably
    extract text + image context.
    """
    prompt_parts = []

    # 1. User instruction
    prompt_parts.append(
        f"""
        USER_QUERY:
        {user_input}
        """.strip()
    )

    # 2. Optional image context
    if images_list:
        image_block = "\n".join(
            [f"- {img}" for img in images_list]
        )
        prompt_parts.append(
            f"""
            IMAGE_INPUTS:
            The user has provided the following image file paths.
            Use them when calling vision/image tools.

            {image_block}
            """.strip()
        )

        prompt_parts.append(
            f"""
            TOOLS_INPUTS:
            For Any Tool which requires request_id you can use the following id

            id - {thread_id}
            """.strip()
        )

    # 3. Tool instruction hint (optional but recommended)
    prompt_parts.append(
        """
        INSTRUCTIONS:
        - If image understanding is required, use the IMAGE_INPUTS.
        - If no images are relevant, answer using text only.
        - Do not assume image content unless explicitly provided.
        """.strip()
    )
    print(prompt_parts)
    return "\n\n".join(prompt_parts)

def validate_request(state: GaurdrailState):
    prompt="""
            Role: You are a request validator for an chatbot.
            
            Task:
            Determine whether the user’s request is allowed based on the approved topics listed below.
            Only evaluate intent and subject matter of the request. Do not answer the request itself.

            Allowed Topics (the request must be primarily about one or more of these):

            Retail images
            Images of stores, shelves, aisles, refrigerators, displays, or product arrangements.
            Shelf count
            Counting shelves, racks, rows, or display levels in a retail image.
            Product count
            Counting how many products, items, units, or packages appear in a retail image.
            Product names inside an image
            Identifying or listing visible product names or brands shown in a retail image.
            Product nutrition
            Nutrition-related information (e.g., calories, ingredients, macros, nutrition labels) for products shown or referenced in a retail image.

            Disallowed Topics (examples):

            Requests unrelated to retail or store images
            Price prediction, demand forecasting, or sales strategy
            Customer behavior analysis
            Personal data or facial recognition
            Medical, legal, or financial advice
            Image manipulation or image generation
            Any topic not clearly connected to the allowed list above

            Output Format (STRICT)

            Respond with only one of the following JSON objects:

            If allowed:
            {
            "allowed": true,
            "reason": "The request is about <brief explanation tied to allowed topics>."
            }

            If disallowed:
            {
            "allowed": false,
            "reason": "The request is not related to retail images, shelf count, product count, product names, or product nutrition."
            }

            Validation Rules
            If the request clearly matches at least one allowed topic, mark it as allowed.
            If the request is ambiguous, but reasonably related to retail images or products, mark it as allowed.
            If the request does not match any allowed topic, mark it as disallowed.
            Do not infer hidden intent beyond what is stated.
        """
    validator_model = llm.with_structured_output(ValidatorSchema)
    response = validator_model.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content=state["user_input"])
                ])
    print("validator response -> ", response)
    if response.allowed:
        return {
            **state,
            "validator_status": "approved",
            "validator_reason": response.reason
        }
    else:
        return {
            **state,
            "validator_status": "rejected",
            "validator_reason": response.reason
        }

def process_flow(state: GaurdrailState):
    thread_id = state["thread_id"]
    images_list = state["images_list"]
    command = state["command"]
    user_input = state["user_input"]

    if user_input is not None:
        save_message(thread_id, "user", user_input)

        prompt = build_prompt(
            user_input=user_input,
            images_list=images_list
        )
    else:
        user_input = command.update["messages"][0].content
        save_message(thread_id, "user", user_input)

    if command:
        input_payload = command
    else:
        input_payload = {
            "messages": [
                HumanMessage(
                    content=prompt,
                    additional_kwargs={"images": images_list or []}
                )
            ],
            "thread_id": thread_id,
        }

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # for message_chunk in graph.stream(
    #     input_payload,
    #     config=config,
    #     stream_mode="values",
    # ):
    #     yield message_chunk
    return {
        **state,
        "__delegate__": {
            "graph": "chat",
            "input": {
                "user_input": state["user_input"],
                "thread_id": state["thread_id"],
                "images_list": state["images_list"],
                "command": state["command"],
                "messages": [
                    HumanMessage(
                        content=prompt,
                        additional_kwargs={"images": images_list or []}
                    )
                ]
            }
        }
    }


def review_router(state: GaurdrailState) -> str:
    return state["validator_status"]

def validator_approved_node(state: GaurdrailState):
    print("approved")
    return {
        **state,
        "messages": [
            AIMessage(content="✅ Validation Approved. Proceeding with the process.\n\n")
        ]
    }

def validator_rejected_node(state: GaurdrailState):
    return {
        "messages": [
            AIMessage(content="❌ Validation rejected. Please ask something related to retail products or product details.")
        ]
    }

def get_validator_graph():

    gaurd_rail_graph = StateGraph(GaurdrailState)

    gaurd_rail_graph.add_node("validate_request",validate_request)
    gaurd_rail_graph.add_node("approved", validator_approved_node)
    gaurd_rail_graph.add_node("rejected", validator_rejected_node)
    gaurd_rail_graph.add_node("process_flow", process_flow)

    gaurd_rail_graph.set_entry_point("validate_request")
    gaurd_rail_graph.add_conditional_edges(
                "validate_request",
                review_router,
                {
                    "approved": "approved",
                    "rejected": "rejected",
                }
            )
    gaurd_rail_graph.add_edge("approved", "process_flow")
    gaurd_rail_graph.add_edge("rejected", END)
    gaurd_rail_graph = gaurd_rail_graph.compile()

    return gaurd_rail_graph
