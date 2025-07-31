"""_summary_
    Take in the transcription and segmented image

Raises:
    Exception: _description_
    Exception: _description_

Returns:
    list: marks of the selected objects
"""

# GPT4 integration
import os
import requests
import base64
from io import BytesIO
from PIL import Image
import json
import re
import numpy as np
# TODO: uncomment following lines
api_key = os.environ["OPENAI_API_KEY"]


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8'), "image/jpeg" if image_path.endswith(".jpg") or image_path.endswith(".jpeg") else "image/png"

# Function to encode the image from PIL object
def encode_image_from_pil(image):
    buffered = BytesIO()
    image_format = image.format.lower()
    image.save(buffered, format=image_format.upper())  # Save in original format
    return base64.b64encode(buffered.getvalue()).decode('utf-8'), f"image/{image_format}"

def decode_base64_image(base64_string, output_path):
    """
    Decode a base64-encoded image and save it to a file for verification.

    Args:
        base64_string (str): The base64-encoded image string (excluding the data URL prefix).
        output_path (str): The path where the decoded image will be saved.
    """
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        
        # Write the binary data to a file
        with open(output_path, "wb") as output_file:
            output_file.write(image_data)
        
        print(f"Image successfully saved to {output_path}")
    except Exception as e:
        print(f"Error decoding image: {e}")


# Read transcription message from a .txt file
def read_transcription(path):
    try:
        with open(path, "r") as file:
            return file.read().strip()  # Remove extra whitespace
    except Exception as e:
        raise Exception(f"Error reading transcription from {path}: {e}")
    
    
# metaprompt = '''
# - Identify the marks of all possible objects that satisfy the user input.
# - For any marks mentioned in your answer, please highlight them with [].
# - Example format: ["11", "12", "13"]
# '''   
metaprompt = """
Identify the relevant objects' mark ids from the segmented image based on the user’s query.  

Response Format (Strict):
- Always return a list of tuples: `[mark1, mark2 , ...]`  

Rules: 
1. If there is a black circle in the image, it indicates potential area of interest.
2. Strictly return only the list (no explanations, no extra text).  
3. If no match is found, return `[]`.

Correct: [11,  12]
Incorrect: `{"marks": ["11", "12"]}`, `["11", "-0.23"]`, `"Best match: 11"` 
"""

def prepare_inputs(message, image):

    # # Path to your image
    # image_path = "temp.jpg"
    # # Getting the base64 string
    if isinstance(image, str):
        base64_image, mime_type = encode_image_from_file(image)
    else:
        base64_image, mime_type = encode_image_from_pil(image)
    payload = {
        "model": "gpt-4o",
        "logprobs": True,
        "messages": [
        {
            "role": "system",
            "content": metaprompt,
            
        }, 
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": message, 
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        
        "max_tokens": 100, 
        "temperature":0
    }

    return payload

def request_gpt4v(message, image):
    payload = prepare_inputs(message, image)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.text}")
    return response.json()

def extract_numbers_and_logprobs(response_json):
    """
    Extracts numbers (mark IDs) and their corresponding log probabilities from a GPT response.

    Args:
        response_json (dict): The JSON response from GPT containing logprobs.

    Returns:
        dict: A dictionary where keys are mark IDs (int) and values are corresponding log probabilities (float).
    """
    mark_logprob_dict = {}

    try:
        # Extract the logprob content from the response
        logprobs_data = response_json["choices"][0]["logprobs"]["content"]

        for item in logprobs_data:
            token = item["token"]
            logprob = item["logprob"]

            # Check if the token is a number
            if re.fullmatch(r'\d+', token):  # Ensures the token is a numeric value
                mark_id = int(token)  # Convert to integer
                mark_logprob_dict[mark_id] = logprob  # Store in dictionary

    except KeyError:
        print("⚠️ Warning: Missing expected fields in response JSON.")

    return mark_logprob_dict


def language_likelihood(mark, extracted_response):
    """
    Computes language probability for each mark based on extracted GPT response.

    Args:
        mark (list): List of objects containing mark IDs and predicted IoUs.
        extracted_response (dict): Dictionary mapping mark IDs to their log probabilities.

    Returns:
        list: Updated `mark` list with added 'lang_prob' values.
    """

    # Check if extracted response is empty
    if not extracted_response:
        print("⚠️ Warning: No valid extracted response. Assigning zero probabilities.")
        for item in mark:
            item['lang_prob'] = 0
        return mark
    
    for item in mark:
        target_id = item['mark']

        # Assign probability based on extracted log-probabilities
        if target_id in extracted_response:
            log_prob = extracted_response[target_id]  # Get log-probability
            prob = np.round(np.exp(log_prob), 6)  # Convert log-prob to probability
        else:
            prob = 0  # Default probability if not in response

        # Ensure probability is within [0, 1]
        item['lang_prob'] = min(max(prob, 0), 1)

    return mark

def run_SoM(image, marks, message = ".tmp/.transcriptions.txt"): 
    if message.endswith(".txt"):
        
        message = read_transcription(message)
    else:
        message = message  # Assume direct transcription message
    print(message)
    try:
        response = request_gpt4v(message, image)
    except Exception as e:
        print("Error:", e)
    
    extracted_resp = extract_numbers_and_logprobs(response)
    mark = language_likelihood(marks, extracted_resp)
    
    return extracted_resp, mark
        

if __name__ == "__main__":
    # # Path to your image
    image_path = "./.tmp/annotated_image.png"
    
    # transcription_path = "speech_recog/.transcriptions.txt"
    # transcription = "I want the red cup. "
    with open('./.tmp/detection_confidence.json', 'r') as file:
        marks = json.load(file)
    run_SoM(image_path, marks)