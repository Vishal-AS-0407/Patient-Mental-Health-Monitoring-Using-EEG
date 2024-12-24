import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyAjhjE1-c6vcFixyO6lOIHQUE8a15peRd0")

model = genai.GenerativeModel("gemini-1.5-flash")

def get_user_input():
    user_info = {}

    user_info['arousal'] = input("Enter Arousal (high/low): ").strip().lower()
    user_info['dominance'] = input("Enter Dominance (high/low): ").strip().lower()
    user_info['valence'] = input("Enter Valence (high/low): ").strip().lower()
    user_info['erp'] = input("Enter ERP (0 or 1): ").strip()

    # User personal information
    user_info['name'] = input("Enter your name: ").strip()
    user_info['age'] = int(input("Enter your age: ").strip())
    user_info['gender'] = input("Enter your gender (Male/Female): ").strip().lower()
    user_info['workplace'] = input("Enter your workplace: ").strip()

    # Trauma and emotional events
    user_info['trauma'] = input("Have you experienced any trauma in the last 2 months? (Yes/No): ").strip().lower()
    if user_info['trauma'] == "yes":
        user_info['trauma_details'] = input("Please provide details of the trauma: ").strip()
    else:
        user_info['trauma_details'] = "None"

    user_info['emotional_breakdown'] = input("Have you experienced any emotional breakdown in the last 2 months? (Yes/No): ").strip().lower()
    if user_info['emotional_breakdown'] == "yes":
        user_info['emotional_breakdown_details'] = input("Please provide details: ").strip()
    else:
        user_info['emotional_breakdown_details'] = "None"

    user_info['positive_events'] = input("Have you experienced any good events in the last 2 months? (Yes/No): ").strip().lower()
    if user_info['positive_events'] == "yes":
        user_info['positive_events_details'] = input("Please provide details of the good events: ").strip()
    else:
        user_info['positive_events_details'] = "None"
    
    return user_info

def generate_input_text(user_info, output_text):
    input_text = f"""
    User Details:
    Name: {user_info['name']}
    Age: {user_info['age']}
    Gender: {user_info['gender']}
    Workplace: {user_info['workplace']}
    
    Arousal: {user_info['arousal']}
    Dominance: {user_info['dominance']}
    Valence: {user_info['valence']}
    ERP: {user_info['erp']}
    
    Trauma Experience: {user_info['trauma_details']}
    Emotional Breakdown: {user_info['emotional_breakdown_details']}
    Positive Events: {user_info['positive_events_details']}
    
    
    ----------------------------------------
    The below text is to get the context under standing of the parameters there may some noise and unwanted data in this and also some of the context may be missing so use this as reference
    {output_text}
    ----------------------------------------

    Please generate a Detailed mental health report analyzing:
    - The impact of Arousal, Dominance, Valence, and ERP on mental health.
    - The emotional state, decision-making skills, and cognitive ability of the user.
    - The user's mental readiness for work/study and any therapy or psychological support recommendations.
    - Suggestions for activities, workload management, and any further suggestions for improving mental health.
    - Any other insights based on the provided data.

    
    """

    return input_text

def generate_report(input_text):
    try:
        response = model.generate_content(input_text)  
        return response.text if response.text else 'No recommendations available'
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Could not generate report."

def save_report(report):
    with open("geminioutput.txt", "w") as file:
        file.write("Mental Health Report\n")
        file.write("====================\n\n")
        file.write(report)
        file.write("\n\nEnd of Report")

def read_output_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading output file: {e}")
        return ""

def main():
    print("Welcome to the Mental Health Report Generator.")
    user_info = get_user_input()

    output_file_path = "biogpt.txt"  
    output_text = read_output_file(output_file_path)  

    if not output_text:
        print("No content found in output.txt. Please ensure the file exists and contains the necessary context.")
        return

    input_text = generate_input_text(user_info, output_text)
    print("Generating the mental health report based on your input...")

    report = generate_report(input_text)
    print("Mental health report generated successfully.")

    save_report(report)
    print("The report has been saved as 'mental_health_report.txt'.")

if __name__ == "__main__":
    main()


