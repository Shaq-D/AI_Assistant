import os
import sqlite3
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL = "gemini-3.5-flash-lite"
DB = "data/ember_restaurant.db"
system = """
You are an system assistant for a restaurant called Ember Resort.
You should provide short and precise answers, in 3 sentences.
If you don't have the answer, don't say anything.
"""
gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=os.getenv("gemini_url")) 

def get_meal_prices(meal): 
    with sqlite3.connect(DB) as conn: 
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM MEALS WHERE meal = ?", (meal.lower(),))
        result = cursor.fetchone()
    return f"The price of {meal} is {result[0]}" if result else None

def get_meal_names(): 
    with sqlite3.connect(DB) as conn: 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM MEALS")
        result = cursor.fetchall()
    return result

price_function = {
    "name": "get_meal_prices", 
    "description": "gets the prices of meals ordered and their names",
    "properties": {
        "type": "object", 
        "parameters": {
            "meal": {
                "type": "string", 
                "description": "the price of a meal ordered by a customer"
            }
        },
        "required": ["meal"],
        "additionalParameters": False
    }
}

name_function = {
    "name": "get_meal_names", 
    "description": "provides the list of meals provided at the restaurant and their prices when asked",
    "properties": {
        "type": "object", 
        "parameters": {
            "meals": {
                "type": "string", 
                "description": "names of meals available"
            }
        },
        "required": ["meals"],
        "additionalParameters": False
    }
}

tools = [{"type": "function", "function": price_function}, {"type": "function", "function": name_function}]

def handleToolCalls(message): 
    responses = []
    for tool_call in message.tool_calls: 
        if(tool_call.function.name == "get_meal_prices"): 
            arguments = json.loads(tool_call.function.arguments)
            meal = arguments.get('meal')
            price = get_meal_prices(meal)
            responses.append({
                "role": "tool", 
                "content": price,
                "tool_call_id": tool_call.id
            })

        if(tool_call.function.name == "get_meal_names"):
            meals = get_meal_names()
            for meal in meals: 
                responses.append({
                    "role": "tool", 
                    "content": meal[0],
                    "tool_call_id": tool_call.id
                })
    return responses

def chat(prompt, history): 
    msg = [] 
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    msg.append(history)
    msg.append(system)
    msg.append(prompt)
    response = gemini.chat.completions.create(model=MODEL, messages=msg, tools=tools)

    while response.choices[0].finish_reason == "tool_calls": 
        message = response.choices[0].message
        rsp = handleToolCalls(message)
        msg.append(message)
        msg.extend(rsp)
        response = gemini.chat.completions.create(model=MODEL, messages=msg, tools=tools)

    reply = response.choices[0].message.content

    history.append({"role": "assistant", "content": reply})

    return history