import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ====================== CONFIGURATION ======================
INPUT_DIR = r"C:\Users\Acer\OneDrive\Desktop\ResearchIntern\Code-Refactoring-QuixBugs-master\python_programs"  
OUTPUT_DIR = r"C:\Users\Acer\OneDrive\Desktop\Code-Refactoring-QuixBugs-master\fixed_programs"
# ===========================================================

# Load environment variables
load_dotenv()

# Initialize models
groq_classifier = ChatGroq(
    temperature=0.1,
    model_name="deepseek-r1-distill-llama-70b",
    groq_api_key="KEY"
)

groq_verifier = ChatGroq(
    temperature=0.1,
    model_name="llama3-70b-8192",
    groq_api_key="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"
)

genai.configure(api_key="KEY")
gemini_llm = genai.GenerativeModel('models/gemini-2.0-flash')

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define defect classes
DEFECT_CLASSES = """1. Off-by-one Error: Example - 'range(len(arr))' instead of 'range(len(arr)-1)'
2. Incorrect Condition: Example - 'if x == y' instead of 'if x != y'
3. Missing Base Case: Example - Recursive function without termination
4. Variable Misassignment: Example - 'total += x' instead of 'total = x'
5. Incorrect Loop Bounds: Example - 'while i < 10' instead of 'while i <= 10'
6. Type Conversion Error: Example - str/int conversion issues
7. Incorrect Operator: Example - Using '+' instead of '*'
8. Missing Returns: Example - Function without return statement
9. Index Error: Example - Accessing arr[len(arr)]
10. Logical Order Error: Example - Swapped operations order
11. Missing Initialization: Example - Using uninitialized variable
12. Incorrect Recursion: Example - Wrong parameters in recursive call
13. Boundary Condition: Example - Not handling empty input
14. Incorrect Formula: Example - Wrong mathematical formula"""

# Agent 1: Deepseek Classifier
cot_classifier_prompt = PromptTemplate(
    input_variables=["code"],
    template=f"""Analyze this code step-by-step to classify the defect:
1. Identify possible defects from: {DEFECT_CLASSES}
2. Compare each possibility to the code
3. Explain your reasoning.Choose the error only from the 14 defect classes.
4. Final answer must be: "<number>. <defect name>"

Code: {{code}}

Reasoning Steps:"""
)
classifier_chain = cot_classifier_prompt | groq_classifier | RunnablePassthrough()

# Agent 2: Gemini Fixer
react_fixer_prompt = PromptTemplate(
    input_variables=["code", "defect_class", "feedback"],
    template="""Fix this Python code using ReAct framework:
Defect: {defect_class}
Previous Feedback: {feedback}

Code:
{code}

Think: Analyze the error and plan the fix
Act: [Write corrected code here]

Return ONLY the code block after "Act:" section. Maintain original formatting.
Should Also Avoid adding any explanations or comments or python written at the top or bottom any unnecessary writing should be avoided at all cost.
Corrected Code:"""
)

def gemini_fixer(inputs):
    response = gemini_llm.generate_content(react_fixer_prompt.format(**inputs))
    return response.text.split("Act:")[-1].strip()

# Agent 3: Llama Verifier
def llama_verifier(original, fixed, defect_class):
    prompt = f"""Verify this code fix:
1. Original Code: {original}
2. Fixed Code: {fixed}
3. Target Defect: {defect_class}(check these as priority)

Analyze differences and potential issues. Check if:
- The fix addresses the reported defect class
- Introduces new errors
- Maintains original functionality
- Follows Python best practices

Return verdict in this format:
VERDICT: [APPROVED/REJECTED]
REASON: <detailed explanation>"""
    
    response = groq_verifier.invoke(prompt)
    return response.content

# Modified repair workflow
def repair_workflow(original_code, max_attempts=7):
    attempts = 0
    current_code = original_code
    defect_class = None
    feedback = "Initial attempt"
    
    print(f"\n{'='*30} Starting Repair Process {'='*30}")
    
    while attempts < max_attempts:
        try:
            print(f"\nAttempt {attempts+1}/{max_attempts}")
            
            # Classify defect with Deepseek
            if attempts == 0:
                print("Classifying defect...")
                defect_response = classifier_chain.invoke({"code": current_code})
                defect_class = defect_response.content.split('\n')[-1].strip()
                print(f"Classification: {defect_class}")
            
            # Generate fix with Gemini
            print("Generating fix...")
            fixed_code = gemini_fixer({
                "code": current_code,
                "defect_class": defect_class,
                "feedback": feedback
            })
            print("Generated fix:\n", fixed_code[:200] + "..." if len(fixed_code) > 200 else fixed_code)
            
            # Verify with Llama
            print("Verifying fix...")
            verification = llama_verifier(original_code, fixed_code, defect_class)
            
            if "VERDICT: APPROVED" in verification:
                print("✓ Fix approved!")
                return fixed_code, "SUCCESS", defect_class
            else:
                feedback = verification.split("REASON: ")[-1] if "REASON: " in verification else verification
                print(f"! Rejection reason: {feedback}")
                current_code = fixed_code
                attempts += 1
                
        except Exception as e:
            print(f"! Error: {str(e)}")
            break
    
    return current_code, "FAILED", f"Max attempts reached. Last feedback: {feedback}"

def process_programs():
    results = []
    
    if not os.listdir(INPUT_DIR):
        print(f"\nError: No Python files found in input directory: {INPUT_DIR}")
        return
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".py"):
            file_path = os.path.join(INPUT_DIR, filename)
            print(f"\n{'='*30} Processing {filename} {'='*30}")
            
            try:
                with open(file_path, "r") as f:
                    original_code = f.read()
                
                fixed_code, status, message = repair_workflow(original_code)
                
                if status == "SUCCESS":
                    fixed_filename = f"{filename}"
                    fixed_path = os.path.join(OUTPUT_DIR, fixed_filename)
                    with open(fixed_path, "w") as f:
                        f.write(fixed_code)
                    print(f"Saved fixed version to: {fixed_path}")
                
                results.append((filename, status, message))
                
            except Exception as e:
                results.append((filename, "ERROR", str(e)))
                print(f"! Processing failed: {str(e)}")
    
    # Print report
    print("\n\n" + "="*50)
    print("{:<30} {:<15} {:<50}".format("Filename", "Status", "Details"))
    print("="*50)
    for filename, status, details in results:
        print("{:<30} {:<15} {:<50}".format(filename, status, details))
    print("="*50)

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"### QuixBugs Repair System - Input: {INPUT_DIR}")
    print(f"###                        Output: {OUTPUT_DIR}")
    print(f"{'#'*60}\n")
    
    process_programs()
    print("\nOperation complete. Verify output directory for results.")
