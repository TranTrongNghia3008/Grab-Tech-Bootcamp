import re
from pathlib import Path
import sys

# Adjust path to import from src
sys.path.append(str(Path(__file__).resolve().parent))

from src.agent.agent_setup import create_agent_executor
from src.config import LOCAL_CHART_DIR
from src.utils.helpers import encode_image_to_base64

def run_chat_loop():
    """Runs the main interactive chat loop for the agent."""
    print("Initializing Database Interaction Agent...")
    try:
        agent_executor = create_agent_executor()
        print("Agent Initialized. Ready for interaction.")
        print(f"Place chart images for analysis in: '{LOCAL_CHART_DIR}'")
        print(f"To analyze a chart, mention its path like: analyze `{LOCAL_CHART_DIR.name}/your_chart.png`")
        print("Type 'exit' or 'quit' to end the session.")
    except Exception as init_err:
        print(f"\n--- Fatal Error During Agent Initialization ---", file=sys.stderr)
        print(f"{init_err}", file=sys.stderr)
        print("Please check your environment variables (especially API keys and DB connection details in .env),", file=sys.stderr)
        print("database status, and required dependencies (`pip install -r requirements.txt`).", file=sys.stderr)
        sys.exit(1)


    while True:
        try:
            user_input_text = input("User: ")
            if user_input_text.lower() in ['exit', 'quit']:
                print("Exiting agent session.")
                break
            if not user_input_text.strip():
                continue

            # --- Multimodal Input Processing ---
            # Simple regex to find paths like `local_chart_images/chart.png` in backticks
            image_path_match = re.search(r"`([^`]+\.(?:png|jpg|jpeg|gif|webp))`", user_input_text)
            agent_input_content = []

            if image_path_match:
                relative_image_path_str = image_path_match.group(1)
                # Construct path relative to the base directory where LOCAL_CHART_DIR resides
                image_path = LOCAL_CHART_DIR.parent / relative_image_path_str # Assume path is relative to project root or includes LOCAL_CHART_DIR name

                # More robust check: ensure it's within the intended directory for security
                if LOCAL_CHART_DIR in image_path.parents or image_path.parent == LOCAL_CHART_DIR:
                    print(f"Attempting to load image: {image_path}")
                    base64_image, mime_type = encode_image_to_base64(image_path)

                    if base64_image and mime_type:
                        agent_input_content.append({"type": "text", "text": user_input_text})
                        agent_input_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            }
                        )
                        print("Image loaded successfully. Sending multimodal input to agent.")
                    else:
                        print(f"Agent: Could not load or encode image: {relative_image_path_str}. Proceeding with text only.")
                        # Fallback to text-only
                        agent_input_content = user_input_text
                else:
                     print(f"Agent: Image path '{relative_image_path_str}' seems outside the allowed '{LOCAL_CHART_DIR}' directory. Ignoring image.")
                     agent_input_content = user_input_text

            else:
                # No image path detected, use text only
                agent_input_content = user_input_text


            # --- Agent Invocation ---
            # The 'input' key matches the placeholder in the prompt
            agent_input = {"input": agent_input_content}

            # Invoke the agent
            response = agent_executor.invoke(agent_input)

            # Print the final response
            print(f"Agent: {response['output']}")


        except KeyboardInterrupt:
             print("\nExiting agent session.")
             break
        except Exception as e:
            print(f"\n--- An error occurred in the chat loop ---", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            print("Continuing chat loop. Please try again or type 'exit'.")
            # Optional: Add logic to reset memory or re-initialize agent on certain errors

if __name__ == "__main__":
    run_chat_loop()