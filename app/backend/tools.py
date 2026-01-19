import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from app.use_case.fetch_shelf_details import FetchShelfDetails
from app.use_case.product_recognition import ProductDetails
from typing import List
from app.use_case.product_as_object_detection import FetchProductAsObjectDetails
from app.use_case.calculate_empty_shelf_percentage import EmptyShelfPercentageDetails
from app.backend.db import log_model_performance

@tool
def calculator( first_num: int, second_num: int, operation: str) -> float:
    """
    Perform operations for given 2 numbers
    Supported operations: add, sub, mul, div

    input fields
        first_num: int
        second_num: int
        operation: str

    response_type
        result: float
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

browser_search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def detect_shelves(image_path: str) -> dict:
    """
    Detect and count shelves in a retail grocery shelf image.

    This tool is used to identify the physical shelf structures
    (horizontal racks/rows) present in a retail or grocery store image.
    It does NOT detect products or product names — only shelf units.

    Typical use cases:
    - Counting the number of shelves in a grocery rack
    - Shelf-level analytics (planogram validation, shelf utilization)
    - Preprocessing step before product or empty-space analysis

    input_fields:
        image_path (str):
            Local file path to a retail shelf image.

    response:
        shelf bounding boxes and metadata
    """
    try:
        print("into detect_shelves")
        start_time = time.time()
        
        request_body = {"file_path": image_path}
        result = FetchShelfDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="YOLO_Shelf_Detection",
                operation="detect_shelves",
                duration_ms=duration_ms,
                input_size=image_path
            )
        except:
            pass  # Don't fail if telemetry fails
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        print("exception is -> ",e)
        return {"status": "error", "message": str(e)}


@tool
def detect_products(image_path: str) -> dict:
    """
    Detect products present in a retail shelf image and return their count.

    This tool detects physical product objects (boxes, bottles, packs, etc.)
    present on shelves. It ONLY detects the presence and bounding boxes
    of products and returns their count.

    Important:
    - This tool does NOT recognize or identify product names or brands.

    Typical use cases:
    - Product density analysis
    - Shelf occupancy measurement
    - Supporting empty-shelf or availability calculations

    input_fields:
        image_path (str):
            Local file path to a retail shelf image.

    response:
        product bounding boxes and count
    """
    try:
        start_time = time.time()
        
        request_body = {"file_path": image_path}
        result = FetchProductAsObjectDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="YOLO_Product_Detection",
                operation="detect_products",
                duration_ms=duration_ms,
                input_size=image_path
            )
        except:
            pass
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def calculate_empty_shelf_percentage(image_path: str) -> dict:
    """
    Calculate the percentage of empty shelf space in a retail shelf image.

    This tool analyzes shelf space versus detected products to determine
    how much shelf area is empty or missing products.

    It helps quantify shelf availability and out-of-stock scenarios.

    Typical use cases:
    - Empty shelf detection
    - Out-of-stock monitoring
    - Retail compliance and merchandising analysis

    input_fields:
        image_path (str):
            Local file path to a retail shelf image.

    response:
        empty percentage, capacity, missing products
    """
    try:
        start_time = time.time()
        
        request_body = {"file_path": image_path}
        result = EmptyShelfPercentageDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="Empty_Shelf_Calculator",
                operation="calculate_empty_percentage",
                duration_ms=duration_ms,
                input_size=image_path
            )
        except:
            pass
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def recognize_products(
    image_paths: List[str],
    request_id: str
) -> dict:
    """
    Recognize and identify products across multiple retail shelf images.

    Unlike detect_products, this tool performs full product recognition.
    It identifies product names / SKUs and aggregates their counts
    across all provided images.

    This tool answers:
    - What products are present?
    - How many times does each product appear across all images?

    Typical use cases:
    - Product recognition and catalog matching
    - Brand / SKU-level shelf analysis
    - Inventory and assortment tracking

    input_fields:
        image_paths (List[str]):
            List of local file paths to retail shelf images.
        request_id (str):
            Unique identifier for logging, tracing, and debugging.

    response:
        product names, confidence scores
    """
    try:
        start_time = time.time()
        
        request_body = {
            "file_paths_list": image_paths,
            "request_id": request_id
        }
        result = ProductDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="SimCLR_Product_Recognition",
                operation="recognize_products",
                duration_ms=duration_ms,
                input_size=f"{len(image_paths)} images"
            )
        except:
            pass
        
        return {
            "status": "success",
            "request_id": request_id,
            "data": result
        }
    except Exception as e:
        print("exception -> ",e)
        return {
            "status": "error",
            "request_id": request_id,
            "message": str(e)
        }


tools_list = [
    calculator, 
    browser_search_tool,
    detect_shelves,
    # detect_products,
    calculate_empty_shelf_percentage,
    recognize_products,
    ]
