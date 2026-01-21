import psycopg
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition


from app.backend.state import ChatState
from app.config import POSTGRES_URI, OPENAI_API_KEY
from app.backend.tools import tools_list
import tiktoken

import tiktoken
import json
from ast import literal_eval
from datetime import datetime
from dotenv import load_dotenv
from app.backend.db import mark_waiting_for_review, clear_waiting_for_review, is_waiting_for_review, create_order, log_model_performance, get_user_age_by_thread
import time
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

load_dotenv()


def count_tokens_and_log(messages, tools):
    """Count and log token usage for debugging"""
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    
    # Create detailed analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "messages": [],
        "tools": [],
        "summary": {}
    }
    
    # Count message tokens with detailed breakdown
    message_tokens = 0
    for i, msg in enumerate(messages):
        content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        msg_tokens = len(encoding.encode(content))
        message_tokens += msg_tokens
        
        msg_analysis = {
            "index": i,
            "type": type(msg).__name__,
            "role": getattr(msg, 'role', 'unknown') if hasattr(msg, 'role') else 'unknown',
            "tokens": msg_tokens,
            "character_count": len(content),
            "content": content
        }
        analysis["messages"].append(msg_analysis)
    
    # Count tool tokens
    tool_tokens = 0
    for i, tool in enumerate(tools):
        tool_str = str(tool)
        individual_tokens = len(encoding.encode(tool_str))
        tool_tokens += individual_tokens
        
        tool_analysis = {
            "index": i,
            "name": getattr(tool, 'name', 'unknown'),
            "tokens": individual_tokens,
            "character_count": len(tool_str),
            "content": tool_str
        }
        analysis["tools"].append(tool_analysis)
    
    total_tokens = message_tokens + tool_tokens
    
    analysis["summary"] = {
        "message_tokens": message_tokens,
        "tool_tokens": tool_tokens,
        "total_tokens": total_tokens,
        "limit": 16385,
        "over_limit": total_tokens > 16385
    }
    
    # Save to file
    debug_file = f"token_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"🔍 TOKEN ANALYSIS:")
    print(f"  📝 Message tokens: {message_tokens}")
    print(f"  🔧 Tool tokens: {tool_tokens}")
    print(f"  📊 Total tokens: {total_tokens}")
    print(f"  Limit: 16,385 tokens")
    print(f"  {'OVER LIMIT' if total_tokens > 16385 else 'Within limit'}")
    print(f"  📄 Detailed analysis saved to: {debug_file}")
    
    return total_tokens

WARNING_THRESHOLDS = {
    "Alcohol": 0,
    "Candy": 2,
    "Canned Food": 5,
    "Chocolate": 2,
    "Dessert": 3,
    "Dried Food": 4,
    "Dried Fruit": 3,
    "Drink": 4,
    "Gum": 5,
    "Instant Drink": 2,
    "Instant Noodles": 1,
    "Milk": 3,
    "Personal Hygiene": 5,
    "Puffed Food": 1,
    "Seasoner": 4,
    "Stationery": 10,
    "Tissue": 3
}


# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-5-nano",
    # model="gpt-3.5-turbo-0125",
    # model = "gpt-4-turbo",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# -----------------------------
# Chat Node
# -----------------------------

# def chatbot_node(state: ChatState):
#     total_tokens = count_tokens_and_log(state["messages"], tools_list)
    
#     if total_tokens > 16385:
#         print("⚠️ WARNING: Token count exceeds limit!")

#     tools_binded_model = llm.bind_tools(tools_list)
#     print("state['messages']-> ",state["messages"])
#     response = tools_binded_model.invoke(state["messages"])

#     print("response from llm is -> ", response)

#     return {
#         "messages": [response],
#         "tools_done": state.get("tools_done", False)
#     }


def chatbot_node(state: ChatState):
    messages = state["messages"].copy()
    thread_id = state["thread_id"]
    print("thread id is -> ",thread_id)
    SYSTEM_HEALTH_PROMPT = f"""
    You are a health-aware assistant.

    You must produce a response with EXACTLY this structure:

    **Detected Items** \n\n

    - <Category> × <Quantity>

    **Health Warnings** \n\n
    <only include items that need warnings>
    - <Category> (<Quantity>): <warning text>

    Rules:
    - EVERY Category line MUST start with a hyphen (-)
    - Use the hyphen (-) exactly so the output can be rendered as a Streamlit list
    - Always show Detected Items
    - Only show Health Warnings if at least one exists
    - Do NOT warn for non-food items (e.g. Tissue, Stationery, Personal Hygiene)
    - Health warnings must be 1–2 sentences
    - General wellness only
    - No medical advice, diagnosis, or fear-based language
    - Mention moderation or balance

    Do not use bullet points.
    Do not add extra sections.

    """
    print("\n")
    print("messages in chatbot_node -> ", messages)
    print("\n")


    if state.get("detected_items") and state.get("tools_done"):
        messages.insert(0, SystemMessage(content=SYSTEM_HEALTH_PROMPT))

        messages.append(
            HumanMessage(
                content=f"""
Detected items:
{state["detected_items"]}

Items requiring health warnings:
{state.get("health_warning_input", [])}
"""
            )
        )

    # Track LLM performance
    llm_start_time = time.time()
    response = llm.bind_tools(tools_list).invoke(messages)
    llm_end_time = time.time()
    
    # Log LLM performance
    llm_duration_ms = (llm_end_time - llm_start_time) * 1000
    try:
        log_model_performance(
            model_name="gpt_llm",
            duration_ms=llm_duration_ms,
            operation="chat_completion",
            input_size=str(len(messages)),
            thread_id=state.get("thread_id"),
            metadata={
                "model": "gpt-5-nano",
                "message_count": len(messages),
                "has_tools": True
            }
        )
    except Exception as e:
        print(f"Failed to log LLM performance: {e}")

    return {
        "messages": [response],
        "tools_done": state.get("tools_done", False)
    }


def review_interrupt_node(state: ChatState):
    thread_id = state["thread_id"]
    # Mark DB state
    mark_waiting_for_review(thread_id)
    print("starting interrupt")
    # Interrupt execution and wait for user input
    user_feedback_query = "Do you want me to proceed with this action? (yes/no)"
    interrupt(
        {
            "question": user_feedback_query
        }
    )
    print("Done interrupt")

def review_decision_node(state: ChatState):
    print("into review decision")
    print(state["messages"])
    last_message = state["messages"][-1].content.lower().strip()
    thread_id = state["thread_id"]

    clear_waiting_for_review(thread_id)

    if "yes" in last_message:
        return {"decision": "approved"}
    elif "no" in last_message:
        return {"decision": "rejected"}
    else:
        return {"decision": "rejected"}  # Default to rejected if unclear

def approved_node(state: ChatState):
    print("approved")
    return {
        "messages": [
            AIMessage(content="Order Creation Requested .")
        ],
        "tools_done": False
    }

def rejected_node(state: ChatState):
    return {
        "messages": [
            AIMessage(content="Order Creation Cancelled.")
        ],
        "tools_done": False
    }

def review_router(state: dict):
    return state["decision"]

def chatbot_router(state: ChatState):
    print("\n")
    print("state in chatbot router -> ", state)
    print("\n")
    tool_route = tools_condition(state)

    if tool_route:
        # Case 1: plain string
        if isinstance(tool_route, str):
            if tool_route == "tools":
                return "tools"

        # Case 2: tuple or list (take first element)
        if isinstance(tool_route, (list, tuple)) and tool_route:
            first = tool_route[0]
            if isinstance(first, str) and first == "tools":
                return "tools"

            # Duck-type Send without importing it
            if hasattr(first, "to") and first.to == "tools":
                return "tools"

        # Case 3: duck-type Send directly
        if hasattr(tool_route, "to") and tool_route.to == "tools":
            return "tools"

    # --- Review after tools ---
    if state.get("tools_done", False) and not is_waiting_for_review(state["thread_id"]):
        return "needs_review"

    print("Going to end")
    # --- End ---
    return END

# def tools_done_node(state: ChatState):
#     return {"tools_done": True}

def tools_done_node(state: ChatState):
    warnings_input = []
    detected_items = {}
    print("state['messages']-> ", state["messages"])
    
    # Find the latest tool message
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "recognize_products":
            try:
                payload = json.loads(msg.content)
                
                # Check if we have a successful response with data
                if payload.get("status") == "success" and "data" in payload:
                    data = payload["data"]
                    
                    # Handle different data structures
                    if isinstance(data, dict):
                        # Check for products_count key
                        if "products_count" in data:
                            detected_items = data["products_count"]
                        # Or maybe the data itself is the products count
                        elif all(isinstance(v, (int, float)) for v in data.values()):
                            detected_items = data
                        else:
                            detected_items = data
                    
                    products = detected_items if isinstance(detected_items, dict) else {}
                    
                    for category, qty in products.items():
                        if isinstance(qty, (int, float)):
                            threshold = WARNING_THRESHOLDS.get(category)
                            if threshold is not None and qty > threshold:
                                warnings_input.append({
                                    "category": category,
                                    "quantity": int(qty)
                                })
                elif payload.get("status") == "error":
                    print(f"Tool returned error: {payload.get('message', 'Unknown error')}")
                    
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Error parsing tool message: {e}")
            break
    
    print("warnings_input is -> ", warnings_input)
    return {
        "tools_done": True,
        "detected_items": detected_items, 
        "health_warning_input": warnings_input
    }

def preprocess_node(state: ChatState):
    """
    Reset per-request tool state when a new image arrives.
    """
    messages = state["messages"]

    if messages:
        last = messages[-1]

        # Detect a new image-based user request
        if (
            isinstance(last, HumanMessage)
            and last.additional_kwargs.get("images")
        ):
            state.pop("detected_items", None)
            state.pop("health_warning_input", None)
            state["tools_done"] = False

    return state

def parse_user_order_request(message: str, detected_items: dict) -> dict:
    """
    Use LLM to parse user's free text message and extract requested products and quantities.
    Example: "I want to order 3 bottles of alcohol and one chocolate" -> {'Alcohol': 3, 'Chocolate': 1}
    """
    
    # Known product categories
    known_products = [
        "Alcohol", "Candy", "Canned Food", "Chocolate", "Dessert", 
        "Dried Food", "Dried Fruit", "Drink", "Gum", "Instant Drink", 
        "Instant Noodles", "Milk", "Personal Hygiene", "Puffed Food", 
        "Seasoner", "Stationery", "Tissue"
    ]
    
    system_prompt = f"""You are a product order parser. Extract products and quantities from the user's message.

Available product categories: {', '.join(known_products)}

Rules:
1. Match user's mentioned products to the closest available category (case-insensitive)
2. Extract the quantity for each product (default to 1 if not specified)
3. If user says "all" or "everything" or "yes", return exactly: {{"all": true}}
4. Return ONLY a valid JSON object with product names as keys and quantities as integer values
5. Use the exact category names from the available list (with proper capitalization)
6. If no products are mentioned or the message is unclear, return an empty object: {{}}

Examples:
- "order 3 alcohol and 1 chocolate" -> {{"Alcohol": 3, "Chocolate": 1}}
- "I want two desserts and 5 candies" -> {{"Dessert": 2, "Candy": 5}}
- "give me some milk" -> {{"Milk": 1}}
- "order all items" -> {{"all": true}}
- "yes please" -> {{}}
- "I'd like to get 3 beers and a sweet treat" -> {{"Alcohol": 3, "Dessert": 1}}

Respond with ONLY the JSON object, no explanation."""

    try:
        # Create a simple LLM instance for parsing (low temperature for consistency)
        parser_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        response = parser_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User message: {message}")
        ])
        
        # Parse the JSON response
        response_text = response.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        print(f"LLM parsed order: {parsed}")
        
        # Handle "all" case
        if parsed.get("all") == True:
            return detected_items.copy()
        
        # Ensure all values are integers
        result = {}
        for key, value in parsed.items():
            if key != "all" and isinstance(value, (int, float)):
                result[key] = int(value)
        
        return result
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing order with LLM: {e}")
        # Fallback: return empty dict (will use all detected items)
        return {}

def validate_order_against_detected(requested: dict, detected: dict) -> tuple[bool, str, dict]:
    """
    Validate that requested products don't exceed detected quantities.
    Returns: (is_valid, error_message, validated_products)
    """
    validated_products = {}
    errors = []
    
    # Create lowercase mapping for case-insensitive comparison
    detected_lower = {k.lower(): (k, v) for k, v in detected.items()}
    
    for product, requested_qty in requested.items():
        product_lower = product.lower()
        
        if product_lower not in detected_lower:
            errors.append(f"'{product}' was not detected in the image")
            continue
        
        original_name, available_qty = detected_lower[product_lower]
        
        if requested_qty > available_qty:
            errors.append(f"'{original_name}': requested {requested_qty}, but only {available_qty} detected")
        else:
            validated_products[original_name] = requested_qty
    
    if errors:
        return False, "\n".join(errors), validated_products
    
    return True, "", validated_products


def create_order_node(state: ChatState):
    detected_items = state.get("detected_items", {})
    thread_id = state["thread_id"]
    
    # Get the user's order request message (the message before the confirmation)
    # We need to find the message where user specified what they want to order
    user_order_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            content = msg.content.lower().strip()
            # Skip confirmation messages (yes/no)
            if content not in ['yes', 'no', 'y', 'n']:
                user_order_message = msg.content
                break
    
    print("\n", "detected_items are -> ", detected_items, "\n")
    print("\n", "thread_id is -> ", thread_id, "\n")
    print("User order message is -> ", user_order_message)
    
    # If no detected items, cannot create order
    if not detected_items:
        return {
            "messages": [
                AIMessage(content="❌ Order creation failed. No products were detected from the image.")
            ]
        }
    
    # Parse user's requested products from their message
    requested_products = parse_user_order_request(user_order_message, detected_items)
    
    print("\n", "requested_products are -> ", requested_products, "\n")
    
    # If no specific products requested, use all detected items
    if not requested_products:
        requested_products = detected_items.copy()
        print("No specific products parsed, using all detected items")
    
    # Validate requested products against detected items
    is_valid, error_message, validated_products = validate_order_against_detected(
        requested_products, detected_items
    )
    
    if not is_valid:
        return {
            "messages": [
                AIMessage(content=f"❌ Order creation cancelled. You have requested more products than detected:\n{error_message}\n\nPlease adjust your order quantities.")
            ]
        }
    
    if not validated_products:
        return {
            "messages": [
                AIMessage(content="❌ Order creation cancelled. No valid products found in your request.")
            ]
        }
    
    # Check if alcohol is in requested products and user is a minor
    alcohol_quantity = 0
    for item_name, qty in validated_products.items():
        if item_name.lower() == "alcohol":
            alcohol_quantity = qty
            break
    
    if alcohol_quantity > 0:
        user_age = get_user_age_by_thread(thread_id)
        print("\n", "user_age is -> ", user_age, "\n")
        if user_age < 21:
            # Remove alcohol from the order but allow other items
            validated_products = {k: v for k, v in validated_products.items() if k.lower() != "alcohol"}
            if not validated_products:
                return {
                    "messages": [
                        AIMessage(content="❌ Order creation failed. Alcohol cannot be sold to minors (under 21 years old), and no other products were requested.")
                    ]
                }
            else:
                # Create order without alcohol
                order_id = create_order(user_id=thread_id, products=validated_products)
                return {
                    "messages": [
                        AIMessage(content=f"⚠️ Alcohol removed from order (not available for customers under 21).\n\n✅ Order created successfully with order_id {order_id}.\n\n**Ordered Products:**\n" + 
                                  "\n".join([f"- {name}: {qty}" for name, qty in validated_products.items()]))
                    ]
                }
    
    # Create the order with validated products
    order_id = create_order(user_id=thread_id, products=validated_products)
    
    return {
        "messages": [
            AIMessage(content=f"✅ Order created successfully with order_id {order_id}.\n\n**Ordered Products:**\n" + 
                      "\n".join([f"- {name}: {qty}" for name, qty in validated_products.items()]))
        ]
    }

# -----------------------------
# SINGLETONS
# -----------------------------
_graph = None
_checkpointer = None
_pg_conn = None




def get_graph():
    global _graph, _checkpointer, _pg_conn

    if _graph is None:
        # -----------------------------
        # DB + CHECKPOINTER
        # -----------------------------
        _pg_conn = psycopg.connect(POSTGRES_URI)
        _pg_conn.autocommit = True

        _checkpointer = PostgresSaver(_pg_conn)
        _store = PostgresStore(_pg_conn)
        _store.setup()
        _checkpointer.setup()

        # -----------------------------
        # GRAPH
        # -----------------------------
        graph = StateGraph(ChatState)

        tool_node = ToolNode(tools_list)

        # -----------------------------
        # NODES
        # -----------------------------
        graph.add_node("chatbot", chatbot_node)
        graph.add_node("tools", tool_node)
        graph.add_node("tools_done", tools_done_node)
        graph.add_node("review_interrupt", review_interrupt_node)
        graph.add_node("review_decision", review_decision_node)
        graph.add_node("approved", approved_node)
        graph.add_node("rejected", rejected_node)
        graph.add_node("preprocess", preprocess_node)
        graph.add_node("create_retail_order", create_order_node)


        # -----------------------------
        # ENTRY
        # -----------------------------
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "chatbot")

        # -----------------------------
        # ROUTING (IMPORTANT)
        # -----------------------------
        graph.add_conditional_edges(
            "chatbot",
            chatbot_router,
            {
                "tools": "tools",
                "needs_review": "review_interrupt",
                END: END,
            }
        )

        # -----------------------------
        # TOOLS LOOP
        # -----------------------------
        graph.add_edge("tools", "tools_done")
        graph.add_edge("tools_done", "chatbot")

        # -----------------------------
        # REVIEW FLOW
        # -----------------------------
        graph.add_edge("review_interrupt", "review_decision")

        graph.add_conditional_edges(
            "review_decision",
            review_router,
            {
                "approved": "approved",
                "rejected": "rejected",
            }
        )

        graph.add_edge("approved", "create_retail_order")
        graph.add_edge("create_retail_order", END)
        graph.add_edge("rejected", END)

        # -----------------------------
        # COMPILE
        # -----------------------------
        _graph = graph.compile(checkpointer=_checkpointer)

    return _graph

