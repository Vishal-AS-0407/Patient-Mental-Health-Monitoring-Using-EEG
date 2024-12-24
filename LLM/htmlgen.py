import google.generativeai as genai
from pathlib import Path
import json

def read_file(filename):
    """Read content from the input file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None

def setup_gemini():
    """Configure and get Gemini model."""
    try:
        GOOGLE_API_KEY = "AIzaSyAjhjE1-c6vcFixyO6lOIHQUE8a15peRd0"  # Replace with actual API key
        genai.configure(api_key=GOOGLE_API_KEY)
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        return genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
    except Exception as e:
        print(f"Error setting up Gemini: {str(e)}")
        return None

def generate_html_with_gemini(model, content):
    """Use Gemini to generate HTML from the content."""
    try:
        prompt = """
        Generate a complete HTML document for a mental health report with embedded CSS. Follow these specific instructions:

        1. HTML Structure:
        - Start with proper DOCTYPE and meta tags
        - Use a simple color scheme: 
        - Include viewport meta tag for responsiveness
        - Create a navigation menu at the top with links to each section
        
        2. CSS Requirements (include within <style> tag):
        - Body: Light background 
        - Text: Use system fonts, line height 1.6
        - Sections: White background, rounded corners, shadow
        - Headers: Bold, slightly larger size, blue color
        - Lists: Proper spacing, bullet styling
        - Tables: Clean borders, alternating row colors
        
        3. Content Organization:
        - Place user details in a prominent card at the top
        - Create separate sections for each major category
        - Use tables for parameter data
        - Format lists with proper indentation
        - Add appropriate spacing between sections
        
        4. Specific Sections to Include:
        - Title and Patient Information
        - Emotional and Cognitive Parameters
        - Analysis Section
        - Recommendations
        - Therapy Suggestions
        - Activities and Management
        - Disclaimer
        
        Here's the content to format into this structure:
        
        {content}
        
        Important: Generate only the complete HTML code with all styling included within a <style> tag in the head section. Make sure all sections are clearly separated and easy to read.
        """
        
        response = model.generate_content(prompt.format(content=content))
        return response.text
    except Exception as e:
        print(f"Error generating HTML: {str(e)}")
        return None

def save_html(html_content, output_file):
    """Save the generated HTML to a file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report successfully generated: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving HTML file: {str(e)}")
        return False

def main():
    # Setup
    input_file = "geminioutput.txt"
    output_file = "mental_health_report.html"
    
    # Read input content
    content = read_file(input_file)
    if not content:
        return

    # Setup Gemini model
    model = setup_gemini()
    if not model:
        return

    # Generate and save HTML
    html_content = generate_html_with_gemini(model, content)
    if not html_content:
        return
    
    save_html(html_content, output_file)

if __name__ == "__main__":
    main()