"""
Conversational Farm Assistant Agent
Flask blueprint — /api/chat endpoint powered by Anthropic Claude.
Calls existing ML endpoints internally (no model duplication).
"""

import os, json, logging
from flask import Blueprint, request, jsonify, current_app
import requests as http_requests

agent_bp = Blueprint('agent', __name__)
log = logging.getLogger(__name__)


# ── Claude tool definitions ──

TOOL_DEFS = [
    {
        "name": "get_weather_summary",
        "description": "Fetch 30-day climate summary for a location (temperature, humidity, precipitation averages). Use whenever the user mentions a location or asks about weather conditions for farming.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "recommend_crop",
        "description": "Predict the best crop to plant given soil parameters and climate conditions. Returns a ranked recommendation with confidence scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "N": {"type": "number", "description": "Nitrogen in kg/ha (0–200)"},
                "P": {"type": "number", "description": "Phosphorus in kg/ha (0–200)"},
                "K": {"type": "number", "description": "Potassium in kg/ha (0–200)"},
                "temperature": {"type": "number", "description": "Temperature in °C"},
                "humidity": {"type": "number", "description": "Humidity in %"},
                "ph": {"type": "number", "description": "Soil pH (3.5–9.5)"},
                "rainfall": {"type": "number", "description": "Rainfall in mm"},
                "lat": {"type": "number", "description": "Optional latitude for live weather"},
                "lon": {"type": "number", "description": "Optional longitude for live weather"},
            },
            "required": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        },
    },
    {
        "name": "forecast_yield",
        "description": "Predict crop yield in tonnes per hectare given area, crop type, weather, and farm size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "area_code": {"type": "integer", "description": "Area code from encoders endpoint"},
                "item_code": {"type": "integer", "description": "Crop item code from encoders endpoint"},
                "area_ha": {"type": "number", "description": "Farm area in hectares"},
                "rainfall": {"type": "number", "description": "Average rainfall in mm"},
                "avg_temp": {"type": "number", "description": "Average temperature in °C"},
                "year": {"type": "integer", "description": "Year for projection"},
                "lat": {"type": "number", "description": "Optional latitude for live weather"},
                "lon": {"type": "number", "description": "Optional longitude for live weather"},
            },
            "required": ["area_code", "item_code", "area_ha", "rainfall", "avg_temp", "year"],
        },
    },
]


# ── Tool execution ──

def _get_base_url():
    """Use the same host the request came in on."""
    return f"http://localhost:{os.environ.get('PORT', 5000)}"


def _call_endpoint(path, body):
    url = f"{_get_base_url()}{path}"
    r = http_requests.post(url, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _handle_weather_summary(lat, lon):
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(BASE, 'src'))
    from weather_client import fetch_climate
    data = fetch_climate(lat, lon)
    if data is None:
        return {"error": "Could not fetch weather data"}
    avg_temp = float(data[:, 0].mean())
    avg_hum = float(data[:, 1].mean())
    total_precip = float(data[:, 2].sum())
    return {
        "location": f"{lat}, {lon}",
        "30day_avg_temperature_c": round(avg_temp, 1),
        "30day_avg_humidity_pct": round(avg_hum, 1),
        "30day_total_precipitation_mm": round(total_precip, 1),
    }


def _handle_recommend_crop(**kwargs):
    path = "/predict-crop"
    body = {k: kwargs[k] for k in ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")}
    if kwargs.get("lat") and kwargs.get("lon"):
        body["lat"] = kwargs["lat"]
        body["lon"] = kwargs["lon"]
    try:
        return _call_endpoint(path, body)
    except Exception as e:
        return {"error": str(e)}


def _handle_forecast_yield(**kwargs):
    path = "/predict-yield-raw"
    body = {k: kwargs[k] for k in ("area_code", "item_code", "area_ha", "rainfall", "avg_temp", "year")}
    if kwargs.get("lat") and kwargs.get("lon"):
        body["lat"] = kwargs["lat"]
        body["lon"] = kwargs["lon"]
    try:
        return _call_endpoint(path, body)
    except Exception as e:
        return {"error": str(e)}


TOOL_HANDLERS = {
    "get_weather_summary": _handle_weather_summary,
    "recommend_crop": _handle_recommend_crop,
    "forecast_yield": _handle_forecast_yield,
}


# ── System prompt ──

SYSTEM_PROMPT = """You are an expert agricultural advisor for Indian farming conditions. You help farmers make better decisions using AI-powered models.

Guidelines:
1. Respond in simple, clear language a farmer can understand.
2. Always use metric units (°C, mm, kg/ha, tonnes/hectare).
3. When the user mentions a location but doesn't give coordinates, ask them for their district/town so you can fetch live climate data.
4. If the user asks about crop selection, use the recommend_crop tool. Only guess parameters from context when the user provides them.
5. If the user asks about yield, use forecast_yield. Explain what the numbers mean.
6. If the user asks about weather for farming, use get_weather_summary.
7. When providing recommendations, mention why a crop suits the given conditions.
8. Keep answers concise but actionable — tell the farmer WHAT to do and WHY.
9. If you're unsure about a parameter, state the assumption you're making.
10. NEVER make up tool results. If a tool returns an error, explain it honestly."""  # noqa: E501


# ── Route ──

@agent_bp.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'Request must include "messages" array'}), 400

        messages = data['messages']
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'ANTHROPIC_API_KEY not configured on server'}), 503

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Conversation loop — Claude may call tools multiple times
        response_text = ""
        max_turns = 5

        for turn in range(max_turns):
            log.info("Agent turn %d/%d — %d messages", turn + 1, max_turns, len(messages))

            resp = client.messages.create(
                model="claude-sonnet-5-20250505",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=TOOL_DEFS,
            )

            # Collect text content
            for block in resp.content:
                if block.type == 'text':
                    response_text = block.text

            # Handle tool calls
            tool_calls = [b for b in resp.content if b.type == 'tool_use']
            if not tool_calls:
                # No more tool calls — Claude gave final answer
                break

            # Append assistant message with tool_use blocks
            messages.append({
                "role": "assistant",
                "content": [{"type": tc.type, "id": tc.id, "name": tc.name, "input": tc.input}
                            for tc in tool_calls],
            })

            # Execute each tool and submit results
            for tc in tool_calls:
                handler = TOOL_HANDLERS.get(tc.name)
                if not handler:
                    result = {"error": f"Unknown tool: {tc.name}"}
                else:
                    try:
                        result = handler(**tc.input)
                    except Exception as e:
                        log.exception("Tool %s failed", tc.name)
                        result = {"error": str(e)}

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps(result),
                        }
                    ],
                })
        else:
            # Max turns reached — use whatever text we have
            log.warning("Agent reached max turns (%d)", max_turns)

        return jsonify({"response": response_text or "I couldn't generate a complete response. Please try again."})

    except Exception as e:
        log.exception("Chat endpoint error")
        return jsonify({"error": str(e)}), 500


# Patch: need sys import at module level for weather tool
import sys  # noqa: E402