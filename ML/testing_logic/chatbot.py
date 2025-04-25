import pandas as pd
import os
import io
import uuid # For unique filenames
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --- Langchain Imports for Custom Agent ---
from langchain.agents import AgentExecutor, create_react_agent # Using ReAct agent as an example
from langchain_core.prompts import PromptTemplate # Or pull from hub
from langchain.tools import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool # Base tool class

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# Add your API key validation here...
print("API Key loaded.")

# --- LLM Configuration ---
try:
    llm = ChatOpenAI(temperature=0, model="gpt-4", api_key=api_key)
    print("Using OpenAI LLM.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

# --- Data Loading ---
csv_data = """Date,Region,Rep,Product,Units,SaleAmount
2024-01-15,East,Alice,Laptop,5,5500.00
2024-01-20,West,Bob,Tablet,10,3500.50
2024-02-10,East,Alice,Monitor,8,2400.00
2024-02-18,South,Charlie,Laptop,3,3300.00
2024-03-05,West,Bob,Keyboard,20,1500.75
2024-03-12,East,Alice,Laptop,2,2200.00
2024-04-01,South,Charlie,Tablet,7,2450.35
2024-04-15,West,Bob,Monitor,12,3600.00
2024-04-20,East,David,Keyboard,15,1125.50
"""
try:
    df = pd.read_csv(io.StringIO(csv_data))
    print("Sample CSV data loaded successfully:")
    # print(df.head()) # Keep it concise for example output
except Exception as e:
    print(f"Error loading sample CSV data: {e}")
    df = None

# --- Define the Custom Python Tool ---
class AutoSavingPythonAstREPLTool(PythonAstREPLTool):
    """A Python REPL tool that automatically saves matplotlib plots generated during execution."""
    save_dir: str = "generated_plots" # Directory to save plots

    def _run(self, query: str) -> str:
        """Execute python code and save plots."""
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Get figure numbers *before* running the code
        existing_fig_nums = set(plt.get_fignums())
        tool_result_obj = None # To store the raw result (could be string, Series, etc.)
        tool_output_str = ""   # To store the final string representation
        error_result = ""
        saved_plot_paths = []

        try:
            # Execute the LLM-generated code using the parent class's method
            tool_result_obj  = super()._run(query)
        except Exception as e:
            error_result = f"\nExecution Error: {e}"
            # Continue execution to save any plots created before the error

        # Check for *new* figures created during execution
        current_fig_nums = set(plt.get_fignums())
        new_fig_nums = current_fig_nums - existing_fig_nums

        if new_fig_nums:
            # print(f"[AutoSaveTool] Detected {len(new_fig_nums)} new plot(s).") # Debugging
            for fig_num in new_fig_nums:
                try:
                    fig = plt.figure(fig_num)
                    # Avoid saving empty figures if plt.figure() was called but nothing plotted
                    if fig.get_axes():
                        filename = os.path.join(self.save_dir, f"plot_{uuid.uuid4()}.png")
                        fig.savefig(filename, bbox_inches='tight') # Save the figure
                        saved_plot_paths.append(os.path.abspath(filename))
                        # print(f"[AutoSaveTool] Saved plot: {filename}") # Debugging
                    plt.close(fig)
                except Exception as save_e:
                    print(f"[AutoSaveTool] Error saving/closing plot for figure {fig_num}: {save_e}")
                    # Try closing again if saving failed but figure exists
                    if plt.fignum_exists(fig_num):
                        try: plt.close(plt.figure(fig_num))
                        except: pass # Ignore errors during final close attempt
               
        # --- Construct the final output string ---

        # 1. Convert the execution result (if any) to string
        if tool_result_obj is not None:
            tool_output_str = str(tool_result_obj)

        # 2. Start building the final result string
        final_result = tool_output_str

        # 3. Append plot message if plots were saved
        if saved_plot_paths:
            plot_message = "\nAutomatically saved plot(s): " + ", ".join([os.path.basename(p) for p in saved_plot_paths])
            # Ensure appending to a string
            final_result = str(final_result) + plot_message

        # 4. Append error message if an error occurred
        if error_result:
             # Ensure appending to a string
            final_result = str(final_result) + error_result

        # 5. Handle cases with no explicit output string (but maybe plots/errors)
        if not final_result: # Check if the string is still effectively empty
            if not saved_plot_paths and not error_result:
                return "[Executed code with no textual output or plots]"
            else:
                # Plots were saved or errors occurred, return the messages generated
                # If final_result is empty here, it means tool_output_str was empty.
                # The plot/error messages should have been added already if they exist.
                # However, let's rebuild just in case.
                rebuilt_result = ""
                if saved_plot_paths:
                    rebuilt_result += "\nAutomatically saved plot(s): " + ", ".join([os.path.basename(p) for p in saved_plot_paths])
                if error_result:
                    rebuilt_result += error_result
                return rebuilt_result if rebuilt_result else "[Code executed, plots/errors handled]"

        return final_result

    # --- Keep your _arun method as is ---
    async def _arun(self, query: str) -> str:
        # Your existing async wrapper logic here
        return self._run(query)


# --- Manual Agent Creation ---
agent_executor = None
if llm and df is not None:
    try:
        # 1. Instantiate the custom tool, providing the DataFrame via locals
        #    (The tool's execution environment needs access to 'df')
        python_tool_instance = AutoSavingPythonAstREPLTool(locals={"df": df})

        # 2. Wrap the custom tool instance for LangChain Agent
        tools = [
            Tool(
                name="python_repl_dataframe", # Needs a descriptive name the LLM can understand
                description="Use this tool to execute Python code on the loaded Pandas DataFrame 'df'. You can use it for data analysis, calculations, manipulation, and plotting. Plots generated will be saved automatically.",
                func=python_tool_instance._run, # Sync execution function
                coroutine=python_tool_instance._arun, # Async execution function
            )
        ]

        # 3. Define the Agent Prompt
        #    We need a prompt that tells the LLM how to use the tool.
        #    Using a standard ReAct prompt structure here. You might need to adapt it.
        #    Ensure the prompt mentions the DataFrame 'df' and the tool's purpose.
        prompt_template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (valid Python code to execute)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        You are working with a pandas DataFrame named 'df'. Here are the first 5 rows:
        {df_head}

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """

        # Get the first 5 rows of the DataFrame as a string for the prompt
        df_head_str = df.head().to_string()

        prompt = PromptTemplate.from_template(prompt_template).partial(df_head=df_head_str)
        # print("Prompt Template Input Variables:", prompt.input_variables) # Debugging

        # 4. Create the Agent using the LLM, tools, and prompt
        agent = create_react_agent(llm, tools, prompt)

        # 5. Create the Agent Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, # Essential for debugging
            handle_parsing_errors=True, # Recommended for robustness
            max_iterations=5 # Prevent runaway agents
        )
        print("\nAgent Executor with custom auto-saving tool created successfully.")

    except Exception as e:
        print(f"\nError creating custom agent: {e}")
        import traceback
        traceback.print_exc()


# --- Interaction Loop (using the custom agent) ---
if agent_executor:
    # User query *without* asking to save
    query = "What is the total SaleAmount per Product? Then create a pie chart and a bar plot for it."

    print(f"\nInvoking agent with query: '{query}'")
    try:
        response = agent_executor.invoke({"input": query})

        print("\nAgent Final Response:")
        print(response.get('output', 'No output key found')) # Agent output structure might vary

        # You don't need to check os.path.exists here explicitly for the user,
        # because the tool's output *should* confirm if plots were saved.
        # But you can check the 'generated_plots' directory manually.
        print(f"\nCheck the '{python_tool_instance.save_dir}/' directory for any saved plots.")

    except Exception as e:
        print(f"\nError during agent invocation: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\nAgent executor not initialized. Cannot process query.")