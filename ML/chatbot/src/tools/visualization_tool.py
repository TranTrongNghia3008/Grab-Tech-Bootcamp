import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from src.database.connection import engine # Import shared engine
from src.config import PLOT_DIR, DEFAULT_FIGSIZE, PIE_CHART_MAX_SLICES

def visualize_data(sql_query: str, chart_type: str, x_col: str, y_col: str = None, color_col: str = None) -> str:
    """
    Executes SQL, fetches data, generates a plot, saves it, and returns the path or error.
    """
    print(f"--- Generating visualization ---")
    print(f"Query: {sql_query}")
    print(f"Chart Type: {chart_type}, X: {x_col}, Y: {y_col}, Color: {color_col}")

    if not sql_query.strip().upper().startswith("SELECT"):
        return "Error: visualize_data tool only works with SELECT queries."

    required_y = ['bar', 'line', 'scatter', 'pie'] # Pie needs a value column too
    if chart_type in required_y and y_col is None:
        return f"Error: 'y_col' is required for chart type '{chart_type}'."

    supported_charts = ['bar', 'line', 'scatter', 'pie', 'hist']
    if chart_type not in supported_charts:
         return f"Error: Unsupported chart type '{chart_type}'. Use one of: {', '.join(supported_charts)}"

    try:
        with engine.connect() as conn:
            # Use pandas to read SQL query results directly into DataFrame
            df = pd.read_sql_query(text(sql_query), conn)

        if df.empty:
            return "Error: Query returned no data to plot."

        # Verify required columns exist in the DataFrame
        required_cols = [x_col]
        if y_col and chart_type != 'hist': # y_col used by bar, line, scatter, pie
            required_cols.append(y_col)
        if color_col:
            required_cols.append(color_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: The following required columns are missing from the query result: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}"

        # --- Plotting Logic ---
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        sns.set_theme(style="whitegrid")

        try:
            if chart_type == 'bar':
                sns.barplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
                ax.set_title(f'Bar Chart: {y_col} by {x_col}')
            elif chart_type == 'line':
                sns.lineplot(data=df, x=x_col, y=y_col, hue=color_col, marker='o', ax=ax)
                ax.set_title(f'Line Chart: {y_col} over {x_col}')
            elif chart_type == 'scatter':
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
                ax.set_title(f'Scatter Plot: {y_col} vs {x_col}')
            elif chart_type == 'pie':
                # Ensure y_col is numeric and x_col has manageable unique values
                if not pd.api.types.is_numeric_dtype(df[y_col]):
                    return f"Error: Column '{y_col}' must be numeric for pie chart values."
                if df[x_col].nunique() > PIE_CHART_MAX_SLICES:
                    # Aggregate smaller slices into 'Other' category if needed, or return error
                    # For simplicity, returning error for now
                    return f"Error: Too many unique values ({df[x_col].nunique()}) in '{x_col}' for a pie chart (max {PIE_CHART_MAX_SLICES}). Consider aggregating data in the SQL query."

                # Use pandas plotting for pie chart directly on the axis
                # Group by x_col and sum y_col if necessary (if SQL didn't aggregate)
                # Assuming SQL provides aggregated data:
                pie_data = df.set_index(x_col)[y_col]
                pie_data.plot(kind='pie', autopct='%1.1f%%', startangle=90, counterclock=False, ax=ax, legend=False)
                ax.set_ylabel('') # Hide y-label for pie
                ax.set_title(f'Pie Chart: Distribution of {y_col} by {x_col}')

            elif chart_type == 'hist':
                if not pd.api.types.is_numeric_dtype(df[x_col]):
                    return f"Error: Column '{x_col}' must be numeric for a histogram."
                sns.histplot(data=df, x=x_col, hue=color_col, kde=True, ax=ax)
                ax.set_title(f'Histogram of {x_col}')

            # Common plot adjustments
            ax.set_xlabel(x_col)
            if chart_type in ['bar', 'line', 'scatter']:
                ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save plot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # Sanitize column names for filename
            safe_x = "".join(c if c.isalnum() else "_" for c in x_col)
            safe_y = "_" + "".join(c if c.isalnum() else "_" for c in y_col) if y_col else ""
            filename = f"plot_{chart_type}_{safe_x}{safe_y}_{timestamp}.png"
            save_path = PLOT_DIR / filename
            plt.savefig(save_path)

            print(f"Plot saved to: {save_path}")
            return f"Plot generated successfully and saved to: {save_path}"

        except Exception as plot_err:
            # Catch errors during the plotting process itself
            return f"Error generating plot visualization: {plot_err}"
        finally:
             # Ensure plot is closed regardless of success or failure inside the try block
             plt.close(fig)

    except (SQLAlchemyError, ProgrammingError) as db_err:
        return f"Error executing SQL query for visualization: {db_err}"
    except KeyError as key_err:
         # This should be caught earlier, but as a fallback
        return f"Error: Column not found in query results after fetching: {key_err}"
    except Exception as e:
        # Ensure plot is closed if error happens before 'finally'
        plt.close() # Try closing the default context if fig wasn't created
        return f"An unexpected error occurred during visualization setup: {e}"