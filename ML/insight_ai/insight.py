import os
import json
from openai import OpenAI
from dotenv import load_dotenv # Import dotenv
import base64

load_dotenv()
# --- Example Data ---

# Example Summary Statistics (as before)
summary_stats_data = [
    {"column": "Store ID", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 25.5, "std": 14.43, "min": 1, "25%": 13, "50%": 25.5, "75%": 38, "max": 50 },
    {"column": "Employee Number", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 53.24, "std": 27.63, "min": 5, "25%": 26, "50%": 57, "75%": 75, "max": 105 },
    {"column": "Area", "count": 3000, "unique": 4, "top": "South America", "freq": 1200, "mean": "", "std": "", "min": "", "25%": "", "50%": "", "75%": "", "max": "" },
    {"column": "Date", "count": 3000, "unique": 60, "top": "2018-01-31", "freq": 50, "mean": "", "std": "", "min": "", "25%": "", "50%": "", "75%": "", "max": "" },
    {"column": "Sales", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 255407.72, "std": 96594.68, "min": 80100.87, "25%": 190691.77, "50%": 236055.14, "75%": 304215.94, "max": 626258.76 },
    {"column": "Marketing Spend", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 14838.92, "std": 8575.66, "min": 6.81, "25%": 7668.02, "50%": 14887.44, "75%": 22103.07, "max": 29991.41 },
    {"column": "Electronics Sales", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 63532.22, "std": 25373.04, "min": 16537.43, "25%": 46076.34, "50%": 58194.94, "75%": 76648.92, "max": 184135.46 },
    {"column": "Home Sales", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 38265.33, "std": 16782.26, "min": 9485.97, "25%": 26431.67, "50%": 34967.10, "75%": 46089.33, "max": 113280.39 },
    {"column": "Clothes Sales", "count": 3000, "unique": "", "top": "", "freq": "", "mean": 89430.01, "std": 35323.99, "min": 24084.93, "25%": 65323.19, "50%": 82971.28, "75%": 106686.96, "max": 233439.01 }
]

# Example Correlation Matrix
correlation_matrix_data = {
    "Store ID": {"Store ID": 1, "Employee Number": -0.137, "Sales": 0.007, "Marketing Spend": 0.032, "Electronics Sales": 0.010, "Home Sales": 0.007, "Clothes Sales": 0.005 },
    "Employee Number": {"Store ID": -0.137, "Employee Number": 1, "Sales": 0.002, "Marketing Spend": -0.003, "Electronics Sales": -0.008, "Home Sales": -0.004, "Clothes Sales": -0.001 },
    "Sales": {"Store ID": 0.007, "Employee Number": 0.002, "Sales": 1, "Marketing Spend": 0.005, "Electronics Sales": 0.939, "Home Sales": 0.870, "Clothes Sales": 0.965 },
    "Marketing Spend": {"Store ID": 0.032, "Employee Number": -0.003, "Sales": 0.005, "Marketing Spend": 1, "Electronics Sales": 0.003, "Home Sales": -0.003, "Clothes Sales": -0.002 },
    "Electronics Sales": {"Store ID": 0.010, "Employee Number": -0.008, "Sales": 0.939, "Marketing Spend": 0.003, "Electronics Sales": 1, "Home Sales": 0.818, "Clothes Sales": 0.905 },
    "Home Sales": {"Store ID": 0.007, "Employee Number": -0.004, "Sales": 0.870, "Marketing Spend": -0.003, "Electronics Sales": 0.818, "Home Sales": 1, "Clothes Sales": 0.840 },
    "Clothes Sales": {"Store ID": 0.005, "Employee Number": -0.001, "Sales": 0.965, "Marketing Spend": -0.002, "Electronics Sales": 0.905, "Home Sales": 0.840, "Clothes Sales": 1 }
}

model_performance_data = {
    "columns": [
        "index", # Model ID
        "Model", # Model Name
        "MAE",   # Mean Absolute Error (Lower is better)
        "MSE",   # Mean Squared Error (Lower is better)
        "RMSE",  # Root Mean Squared Error (Lower is better)
        "R2",    # R-squared (Closer to 1 is better)
        "RMSLE", # Root Mean Squared Log Error (Lower is better)
        "MAPE",  # Mean Absolute Percentage Error (Lower is better)
        "TT (Sec)" # Training Time in Seconds (Lower is better)
    ],
    "data": [
        ["ridge", "Ridge Regression", 7012.6336, 81087526.7735, 9001.7974, 0.8755, 0.1316, 0.1118, 0.95],
        ["lr", "Linear Regression", 7012.7219, 81089281.0402, 9001.8958, 0.8755, 0.1316, 0.1118, 1.688],
        ["rf", "Random Forest Regressor", 7286.7424, 89825481.376, 9470.0525, 0.8623, 0.1381, 0.1163, 0.192],
        ["et", "Extra Trees Regressor", 7407.6405, 92479944.746, 9608.7379, 0.858, 0.14, 0.1184, 0.138],
        ["lightgbm", "Light Gradient Boosting Machine", 7429.7621, 96262849.9472, 9807.0619, 0.8522, 0.1413, 0.1179, 0.702]
    ]
}

# Example Tuned Model Results
tuned_model_results_data = {
    "best_params": {
        "copy_X": True, "fit_intercept": True, "n_jobs": -1, "positive": False
    },
    "cv_metrics_table": {
        "columns": ["Fold", "MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"],
        "data": [
            [0, 6810.6922, 78120927.356, 8838.6044, 0.8753, 0.131, 0.1099],
            [1, 7170.3902, 87006939.3146, 9327.751, 0.8797, 0.13, 0.1112],
            [2, 6553.2886, 72443314.1277, 8511.3638, 0.8662, 0.1238, 0.1036],
            [3, 6574.8354, 69304606.1913, 8324.9388, 0.9053, 0.1269, 0.1067],
            [4, 6917.6908, 79112653.5021, 8894.5294, 0.8725, 0.1298, 0.111],
            ["Mean", 6805.3795, 77197688.0983, 8779.4375, 0.8798, 0.1283, 0.1085],
            ["Std", 229.1616, 6093368.3048, 345.2035, 0.0135, 0.0026, 0.0029]
        ]
    }
}

# This would be the actual path to your image file if running locally,
# or a public URL if the image is hosted.
# For base64 encoding, you would read the file and encode it.
feature_importance_image_path  = "../automl/feature_importance.png" # Or "https://example.com/path/to/your/image.png"

# --- Prompts ---
def get_model_performance_prompt(perf_string):
    """Returns the prompt for summarizing model performance results."""
    return f"""
    Analyze the following machine learning model performance comparison table. The table structure is defined by 'columns' and model results are in 'data'. Metrics might be for regression (like MAE, MSE, RMSE, R2 - lower error is better, R2 closer to 1 is better) or classification (like Accuracy, Precision, Recall, F1, AUC - generally higher is better). 'TT (Sec)' is training time in seconds (lower is better).

    ```json
    {perf_string}
    ```

    Provide a **concise summary** for a non-technical or semi-technical audience, explaining the key findings and recommending next steps.

    **Instructions for Analysis:**
    1.  **Identify Top Performers:** Determine which 1-2 models perform best based on the primary performance metrics present (e.g., lowest RMSE/MAE, highest R2 for regression; highest Accuracy/F1/AUC for classification). Mention the key metric(s) used for comparison.
    2.  **Consider Performance vs. Speed Trade-off:**
        * Highlight any models that offer a good balance (e.g., slightly lower performance but significantly faster training time `TT (Sec)`).
        * Point out if the top-performing models are particularly slow to train.
    3.  **Note Performance Tiers/Similarities:** Are there groups of models with very similar performance? Are simpler models (like Linear/Ridge Regression) competitive with more complex ones (like Random Forest, Gradient Boosting)?
    4.  **Mention Metric Consistency (if applicable):** Do different key metrics generally agree on the best model(s)? If a model excels in one metric but performs poorly in another relevant one, briefly note it.
    5.  **Be Concise and Clear:** Avoid overly technical jargon. Explain results in terms of "better accuracy," "lower prediction error," "faster training," etc.

    **Instructions for Recommendations:**
    **Propose Actionable Next Steps:** use a recommend and friendly tone to suggest concrete actions such as:
        * Suggest Candidates for Deployment/Further Work: Based on performance and speed, recommend 1-2 models as strong candidates for potential deployment or more intensive tuning.
        * "Further tune the hyperparameters of [Top Model Name(s)] to potentially improve performance."
        * "Investigate why [Specific Model] performs surprisingly well/poorly compared to others."
        * "Consider [Model Name] if training speed is a critical factor, despite slightly lower performance."
        * "Evaluate models based on the specific business metric that matters most before making a final decision."
        * (If performance is poor overall): "Consider exploring different feature engineering techniques or model architectures."

    **Output Format:**
    * A brief introductory sentence about comparing model performance.
    * Key findings regarding top performers and trade-offs (bullet points preferred).
    * Actionable recommendations and next steps (bullet points preferred).
    """

def get_summary_stats_prompt(stats_string):
    """Returns the prompt for summarizing descriptive statistics."""
    return f"""
    Analyze the following statistical summary data derived from a dataset:

    ```json
    {stats_string}
    ```

    Provide a **concise summary** (around 2-3 key bullet points or a short paragraph) for a non-technical person.

    **Instructions:**
    1.  **Be Brief:** Do not simply list all statistics for every column.
    2.  **Highlight Key Insights:** Focus on what's most *interesting* or *significant*. Look for:
        * **Dominant Categories:** Is one category ('top') much more frequent ('freq') than others (like in 'Area')?
        * **Wide Variations:** Are the numbers spread out? (e.g., large difference between 'min'/'max', or a high 'std' relative to the 'mean' for things like Sales or Marketing Spend). Explain this simply (e.g., "Sales figures vary quite a lot").
        * **Typical Values:** Briefly mention averages ('mean') or middle values ('50%') for important numerical columns like Sales.
        * **Anything Unusual:** Point out anything that seems noteworthy (e.g., very low minimum Marketing Spend).
    3.  **Suggest Next Steps:** use a recommend and friendly tone and based *only* on this summary, propose 1-2 specific, actionable questions or areas for further investigation that arise from the highlighted insights. Frame these as logical next steps someone might take. Examples: "Investigate why South America is the dominant area," "Explore the reasons behind the wide range in Sales," "Analyze the relationship between Marketing Spend and Sales."

    **Output Format:**
    * A brief introductory sentence.
    * Key insights (bullet points preferred).
    * Suggested and recommendation for next steps/questions (bullet points preferred).

    Keep the language simple and clear.
    """

# --- REFINED CORRELATION PROMPT ---
def get_correlation_matrix_prompt(corr_string):
    """Returns the prompt for summarizing a correlation matrix, focusing on non-obvious insights."""
    return f"""
    Analyze the following correlation matrix. This matrix shows how different numerical variables tend to move together (correlation coefficient from -1 to +1).

    ```json
    {corr_string}
    ```

    Provide a **concise summary** for a non-technical person, focusing on **genuinely interesting or unexpected findings** and suggesting actionable next steps. Avoid stating the obvious.

    **Instructions for Identifying Insights:**
    1.  **Prioritize Non-Obvious Strong Correlations:** Focus on strong positive (> 0.6 or 0.7) or strong negative (< -0.6 or -0.7) correlations between variables that represent **distinct concepts**.
        * *Do not mention* highly expected strong correlations. These are less insightful.
        * *Highlight* strong correlations between variables that are *not* directly related by definition.
    2.  **Highlight Significant Negative Correlations:** These often reveal interesting trade-offs or inverse relationships. Explain the potential implication simply.
    3.  **Identify Surprisingly Weak Correlations:** Point out correlations that are close to zero where a moderate or strong relationship might have been expected based on the variable names (e.g., between a spending variable and an outcome variable, if applicable). This lack of relationship can be as insightful as a strong one.
    4.  **Look for Clusters (Optional):** If multiple distinct variables are all highly correlated with each other, mention this potential grouping or shared underlying factor.
    5.  **Be Selective:** Do not list every correlation. Focus on the 2-4 most insightful findings based on the criteria above. Ignore self-correlations (1.0).

    **Instructions for Explaining and Next Steps:**
    1.  **Explain Simply:** Describe the relationship (e.g., "Variable A and Variable B tend to strongly increase together," or "Variable C tends to decrease when Variable D increases").
    2.  **Suggest Actionable Next Steps:** use a recommend and friendly tone. For each key insight, propose 1-2 specific questions or investigation areas. These should aim to understand *why* the relationship (or lack thereof) exists or what the business implications are. Frame them generally. Examples:
        * "Investigate the underlying reason for the unexpected strong link between [Variable X] and [Variable Y]."
        * "Explore factors that might explain the surprisingly weak relationship between [Variable P] and [Variable Q]."
        * "Analyze the potential trade-off indicated by the negative correlation between [Variable M] and [Variable N]."
        * "Examine if the cluster of correlated variables ([V1], [V2], [V3]) represents a common driver or influence."

    **Output Format:**
    * The most insightful relationships found (bullet points preferred).
    * Suggested next steps/questions based on those insights (bullet points preferred).

    Keep the language simple, clear, and focused on practical, non-obvious insights.
    """

# --- REFINED PROMPT FOR TUNED MODEL & IMAGE (Focus on Insights & Actions) ---
def get_tuned_model_with_image_prompt_text(tuning_results_string):
    """
    Returns the textual part of the prompt for summarizing tuned model results
    and a feature importance image, focusing on insights and actions for a non-technical user.
    This text is intended for a multimodal model like GPT-4o that will also see the image.
    """
    return f"""
    You have two pieces of information about an optimized prediction model:
    1.  **Tuning Performance Data (text below):** Results from testing the model after finding its best settings. This includes average performance and consistency across several tests.
        ```json
        {tuning_results_string}
        ```
    2.  **Feature Importance Plot (image):** An image you will also analyze, showing which input data characteristics (features) the model uses most to make predictions.

    **Your Task for a Non-Technical User:**
    Skip basic definitions. Directly provide valuable and interesting insights from both the tuning data and the feature importance image. Focus on what these findings mean practically and suggest clear, actionable next steps.

    **1. Key Findings from Model's Performance Tests (from JSON data):**
    * **How well did it perform on average?** Look at the 'Mean' row in the `cv_metrics_table`. Highlight a key performance indicator (e.g., for regression, focus on 'RMSE' or 'MAE' as "average prediction error," or 'R2' as "accuracy in explaining outcomes." For classification, it would be 'Accuracy', 'F1', etc.). State the average value simply.
        * *Insight example:* "On average, the model's predictions had an error of about [Mean RMSE value]." or "The model was able to explain about [Mean R2 value * 100]% of the outcomes."
    * **How consistent was its performance?** Look at the 'Std' (Standard Deviation) row for that same key metric.
        * *Insight example:* "This performance was [very consistent / fairly consistent / showed some noticeable variation] across the different tests (consistency score: [Std value])." (A low Std relative to the Mean is more consistent).

    **2. Key Factors Driving the Model's Predictions (from the Feature Importance Image):**
    * **What are the top 2-3 game-changers?** Based on your analysis of the image, identify the most influential features.
        * *Insight example:* "The model relies most heavily on [Top Feature 1] and [Top Feature 2] to make its predictions. [Feature 3] also plays a significant role."
    * **Are there any surprises among these key factors?**
        * *Insight example:* "It's interesting that [Surprising Top Feature] is so important. Does this give you a new perspective?" or "As might be expected, [Obvious Top Feature] is a key driver."
    * **Which factors had little impact?** Briefly mention if any commonly considered factors are shown to be unimportant by the model.
        * *Insight example:* "Interestingly, factors like [Less Important Feature 1] and [Less Important Feature 2] didn't have much influence on the model's predictions in this setup."

    **3. Actionable Recommendations & Next Steps (Synthesize Everything):** use a recommend and friendly tone to: 
    * **Based on Performance:**
        * If good average performance & good consistency: "This optimized model performs well and reliably. It appears ready for [e.g., pilot testing, deployment in a controlled environment, use for specific decisions]."
        * If good average performance & poor consistency: "While the model shows good average performance, its results varied a bit across tests. If high reliability is critical for every prediction, consider [e.g., further investigation into the variability, testing on more diverse data before full deployment]."
        * If performance is borderline or needs improvement: "The current performance is [describe simply]. To improve it, you might consider [e.g., collecting more data on key features, trying different model types, further refining features]."
    * **Leveraging Feature Insights:**
        * "Focus on ensuring the data for the key drivers ([Top Feature 1], [Top Feature 2]) is always high quality and accurate, as they heavily influence the results."
        * "Discuss with your team: How can you use the knowledge that [Top Feature 1] and [Top Feature 2] are so influential? Does it change any business strategies or data collection priorities?"
    * **Specific Next Steps to Consider:**
        * "Validate these key features: Confirm with experts if the importance of these features makes sense and if it uncovers anything new."
        * "Monitor performance: If you decide to use this model, keep an eye on how well it performs on new, real-world data."
        * "Iterate if needed: Based on its real-world performance and your business goals, you can always revisit and further refine the model or features."

    Keep your language direct, simple, and focused on practical value.
    """
# --- Combined Function (Updated) ---

# --- Helper function to encode image ---
def encode_image_to_base64(image_path):
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Cannot include image in the API call.")
        return None
    except Exception as e:
        print(f"Warning: Error encoding image {image_path}: {e}")
        return None
# --- Combined Function (Updated) ---
# --- Main Function to Get AI Summary ---
def get_ai_summary(data, input_type, api_key=None, model_name="gpt-4o-mini"):
    """
    Generates an AI summary for various data types, including multimodal inputs.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

    client = OpenAI(api_key=api_key)
    text_prompt_content = None
    system_message = "You are a helpful AI assistant skilled at explaining data insights in simple terms."
    temperature = 0.4
    max_tokens = 500
    final_model_name = model_name

    messages = []

    if input_type == 'summary_stats':
        data_string = json.dumps(data, indent=2)
        text_prompt_content = get_summary_stats_prompt(data_string)
        system_message = "You are a helpful and friendly AI assistant skilled at extracting key insights from statistical summaries and suggesting next steps in simple terms. Format the output in HTML format."
        final_model_name = model_name or "gpt-3.5-turbo"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text_prompt_content}
        ]
    elif input_type == 'correlation_matrix':
        data_string = json.dumps(data, indent=2)
        text_prompt_content = get_correlation_matrix_prompt(data_string)
        system_message = "You are a helpful and friendly AI assistant skilled at explaining correlation matrices, focusing on non-obvious insights and their implications in simple terms. Format the output in HTML format."
        final_model_name = model_name or "gpt-3.5-turbo"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text_prompt_content}
        ]
    elif input_type == 'model_performance':
        data_string = json.dumps(data, indent=2)
        text_prompt_content = get_model_performance_prompt(data_string)
        system_message = "You are a helpful and friendly AI assistant skilled at interpreting machine learning model performance results and providing actionable recommendations in simple terms. Format the output in HTML format."
        final_model_name = model_name or "gpt-3.5-turbo"
        max_tokens = 700
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text_prompt_content}
        ]
    elif input_type == 'tuned_model_with_image_eval':
        if not isinstance(data, dict) or "tuning_data" not in data or "feature_importance_image_path" not in data:
            return "Error: For 'tuned_model_with_image_eval', data must be a dictionary with keys 'tuning_data' and 'feature_importance_image_path'."

        tuning_results_string = json.dumps(data["tuning_data"], indent=2)
        image_path = data["feature_importance_image_path"]
        text_prompt_content = get_tuned_model_with_image_prompt_text(tuning_results_string)
        system_message = "You are an AI assistant skilled at explaining complex model tuning results by synthesizing textual data and visual feature importance plots for a non-technical audience. Format the output in HTML format."
        final_model_name = model_name or "gpt-4o" # Default to GPT-4o for multimodal
        max_tokens = 700 # Allow more for detailed multimodal explanation

        user_content = [{"type": "text", "text": text_prompt_content}]

        # Handle image: URL or local file path
        if image_path.startswith("http://") or image_path.startswith("https://"):
            user_content.append({"type": "image_url", "image_url": {"url": image_path}})
        else: # Assume local file path
            base64_image = encode_image_to_base64(image_path)
            if base64_image:
                # Determine MIME type (basic inference, can be improved)
                mime_type = "image/png" if image_path.lower().endswith(".png") else \
                            "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else \
                            "image/gif" if image_path.lower().endswith(".gif") else \
                            "image/webp" if image_path.lower().endswith(".webp") else \
                            "application/octet-stream" # Fallback
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
            else:
                # If image encoding failed, we can still send the text part or error out
                print(f"Warning: Could not process image at {image_path}. Proceeding with text-only analysis if possible, or this part might be incomplete.")
                # Optionally, you could modify the prompt to inform the AI the image is missing.
                # For now, it will proceed and the AI might comment on the missing image based on the prompt.

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
    else:
        return f"Error: Invalid input_type '{input_type}'."

    if not messages:
         return "Error: Could not construct messages for API call."

    try:
        response = client.chat.completions.create(
            model=final_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during OpenAI API call: {e}"

# --- Main Execution Examples ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. This script requires it to function.")
        exit()
    
    # To run the multimodal example, ensure "Feature Importance.png" is in the same directory as the script,
    # or provide the correct path.
    # You might need to create a dummy "Feature Importance.png" if you don't have the 


    print("="*20 + " Analyzing Summary Statistics " + "="*20)
    summary_stats_result = get_ai_summary(summary_stats_data, input_type='summary_stats')
    print(summary_stats_result)
    print("\n" + "="*60 + "\n")

    print("="*20 + " Analyzing Correlation Matrix " + "="*20)
    correlation_result = get_ai_summary(correlation_matrix_data, input_type='correlation_matrix')
    print(correlation_result)
    print("\n" + "="*60 + "\n")

    print("="*20 + " Analyzing Model Performance " + "="*20)
    model_perf_result = get_ai_summary(model_performance_data, input_type='model_performance')
    print(model_perf_result)
    print("\n" + "="*60 + "\n")

    print("="*20 + " Analyzing Tuned Model & Feature Importance (Multimodal Call) " + "="*20)
    tuned_model_input_payload = {
        "tuning_data": tuned_model_results_data,
        "feature_importance_image_path": feature_importance_image_path # Local path
    }
    tuned_model_summary = get_ai_summary(tuned_model_input_payload, input_type='tuned_model_with_image_eval')
    print(tuned_model_summary)
    print("\n" + "="*60 + "\n")
