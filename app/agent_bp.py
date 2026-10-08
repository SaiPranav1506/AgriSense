"""
Conversational Farm Assistant Agent
Flask blueprint — /api/chat endpoint powered by Anthropic Claude.
Calls existing ML endpoints internally (no model duplication).
"""

import os, json, logging, sys
from flask import Blueprint, request, jsonify, current_app
import requests as http_requests

# Load the project-root .env (NVIDIA_API_KEY / ANTHROPIC_API_KEY etc.).
# The .env file is gitignored, so real keys never reach the repo.
try:
    from dotenv import load_dotenv
    load_dotenv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'
        )
    )
except Exception:  # python-dotenv is optional
    pass

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


# ── Agent providers ────────────────────────────────────────────────────
# The agent runs on NVIDIA NIM (OpenAI-compatible) when NVIDIA_API_KEY is
# set, otherwise on Anthropic Claude. Both paths expose the same tools.

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
NVIDIA_MODEL = os.environ.get('NVIDIA_MODEL', 'meta/llama-3.3-70b-instruct')


def _openai_tools():
    """Translate the Anthropic-style TOOL_DEFS into OpenAI function specs."""
    return [
        {
            'type': 'function',
            'function': {
                'name': t['name'],
                'description': t['description'],
                'parameters': t['input_schema'],
            },
        }
        for t in TOOL_DEFS
    ]


def _run_tool(name, args):
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {'error': f'Unknown tool: {name}'}
    try:
        return handler(**(args or {}))
    except Exception as e:
        log.exception('Tool %s failed', name)
        return {'error': str(e)}


def _last_user_text(messages):
    """Plain-text of the most recent user turn (for demo-mode replies)."""
    for m in reversed(messages):
        if m.get('role') == 'user':
            c = m.get('content', '')
            if isinstance(c, list):
                return ' '.join(
                    b.get('text', '') for b in c
                    if isinstance(b, dict) and b.get('type') == 'text'
                )
            return str(c)
    return ''


def _run_nvidia(messages, max_turns=5):
    """Tool-calling conversation loop against NVIDIA NIM (OpenAI-compatible)."""
    convo = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for m in messages:
        if m.get('role') in ('user', 'assistant'):
            convo.append({'role': m['role'], 'content': m.get('content', '')})

    url = f"{NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        'Authorization': f'Bearer {NVIDIA_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }

    response_text = ''
    for turn in range(max_turns):
        log.info('NVIDIA agent turn %d/%d — %d messages', turn + 1, max_turns, len(convo))
        payload = {
            'model': NVIDIA_MODEL,
            'messages': convo,
            'tools': _openai_tools(),
            'tool_choice': 'auto',
            'temperature': 0.3,
            'max_tokens': 1024,
        }
        r = http_requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f'NVIDIA API {r.status_code}: {r.text[:300]}')

        msg = ((r.json().get('choices') or [{}])[0].get('message')) or {}
        if msg.get('content'):
            response_text = msg['content']

        tool_calls = msg.get('tool_calls') or []
        if not tool_calls:
            break

        convo.append({
            'role': 'assistant',
            'content': msg.get('content') or '',
            'tool_calls': tool_calls,
        })
        for tc in tool_calls:
            fn = tc.get('function') or {}
            try:
                args = json.loads(fn.get('arguments') or '{}')
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(fn.get('name'), args)
            convo.append({
                'role': 'tool',
                'tool_call_id': tc.get('id'),
                'content': json.dumps(result),
            })
    else:
        log.warning('NVIDIA agent reached max turns (%d)', max_turns)

    return response_text


def _run_anthropic(messages, api_key, max_turns=5):
    """Tool-calling conversation loop against Anthropic Claude."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msgs = list(messages)
    response_text = ''

    for turn in range(max_turns):
        log.info('Anthropic agent turn %d/%d — %d messages', turn + 1, max_turns, len(msgs))
        resp = client.messages.create(
            model='claude-sonnet-5-20250505',
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=msgs,
            tools=TOOL_DEFS,
        )
        for block in resp.content:
            if block.type == 'text':
                response_text = block.text

        tool_calls = [b for b in resp.content if b.type == 'tool_use']
        if not tool_calls:
            break

        msgs.append({
            'role': 'assistant',
            'content': [
                {'type': tc.type, 'id': tc.id, 'name': tc.name, 'input': tc.input}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            result = _run_tool(tc.name, tc.input)
            msgs.append({
                'role': 'user',
                'content': [{
                    'type': 'tool_result',
                    'tool_use_id': tc.id,
                    'content': json.dumps(result),
                }],
            })
    else:
        log.warning('Anthropic agent reached max turns (%d)', max_turns)

    return response_text


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
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')

        if not NVIDIA_API_KEY and not anthropic_key:
            return jsonify({
                'error': 'No agent provider configured — set NVIDIA_API_KEY '
                         'or ANTHROPIC_API_KEY on the server'
            }), 503

        try:
            if NVIDIA_API_KEY:
                log.info('Agent provider: NVIDIA (%s)', NVIDIA_MODEL)
                response_text = _run_nvidia(messages)
            else:
                log.info('Agent provider: Anthropic')
                response_text = _run_anthropic(messages, anthropic_key)
        except Exception as provider_err:
            # Key rejected / network down -> graceful demo fallback.
            log.warning('Agent provider failed (%s) -- falling back to demo mode', provider_err)
            user_msg = _last_user_text(messages)
            log.info('DEMO mode reply for: %s', str(user_msg)[:80])
            return jsonify({'response': _demo_reply(user_msg), 'demo_mode': True})

        return jsonify({
            'response': response_text
            or "I couldn't generate a complete response. Please try again."
        })

    except Exception as e:
        log.exception("Chat endpoint error")
        return jsonify({"error": str(e)}), 500





# ---------------------------------------------------------------------------
# Demo mode fallback -- used when the Anthropic API key is missing/rejected
# ---------------------------------------------------------------------------

DEMO_BANNER = (
    "> Note: this is a *demo* reply -- the Anthropic API key on this server "
    "is invalid or out of credit, so I cannot call Claude right now.\n"
    "> Set a valid ANTHROPIC_API_KEY in your environment (or .env) and "
    "restart Flask for full chat."
)


def _demo_reply(user_msg):
    u = (user_msg or '').lower()
    if any(k in u for k in ('crop', 'recommend', 'plant', 'grow', 'soil')):
        body = (
            "**Suggested top crops (demo):**\n\n"
            "1. **Rice** -- thrives in warm, humid conditions (24C+, >70% RH) "
            "with heavy rainfall.\n"
            "2. **Maize** -- adapts to moderate temps (21-27C) and "
            "well-drained loam.\n"
            "3. **Cotton** -- loves high temps (25-35C) and moderate water.\n\n"
            "To get a real ranked recommendation with confidence scores, send "
            "your soil N, P, K, pH, and climate values -- the backend's "
            "/predict-crop endpoint powers it."
        )
    elif any(k in u for k in ('yield', 'forecast', 'production', 'tonnes', 'ton/hectare', 'tons')):
        body = (
            "**Sample yield estimate (demo):**\n\n"
            "- Crop: *Rice, paddy*\n"
            "- Region: *India*\n"
            "- Projected yield: ~3.4 t/ha for normal conditions\n\n"
            "Yield varies a lot with rainfall, fertilizer use, and temperature. "
            "Call /predict-yield with your actual parameters for an accurate "
            "forecast."
        )
    elif any(k in u for k in ('weather', 'rain', 'humidity', 'climate')):
        body = (
            "**Weather guidance (demo):**\n\n"
            "I can pull a 30-day climate summary from Open-Meteo if you share "
            "your latitude and longitude.\n\n"
            "For Indian farming zones right now we typically see:\n"
            "- North India: 18-32C, low humidity\n"
            "- South India: 24-35C, 60-85% humidity\n"
            "- Coastal: high humidity, heavy monsoon rainfall Jul-Sep\n\n"
            "Send your coordinates for live numbers."
        )
    elif any(k in u for k in ('disease', 'pest', 'leaf spot', 'blight', 'infection')):
        body = (
            "**Disease detection (demo):**\n\n"
            "Upload a photo of the affected leaf on the **Disease** page -- "
            "the MobileNetV2 model will classify it and return top-3 "
            "predictions with confidence.\n\n"
            "Common conditions: tomato leaf curl, maize rust, rice bacterial "
            "blight, citrus canker."
        )
    elif any(k in u for k in ('hi', 'hello', 'hey', 'namaste')):
        body = (
            "Hello! I am the AgriSense farm assistant (currently in demo mode).\n\n"
            "You can ask me about:\n"
            "- Crop recommendations -- based on your soil and climate\n"
            "- Yield forecasts -- for your crop, region, and farm size\n"
            "- Weather summaries -- live climate for your location\n"
            "- Disease detection -- upload a leaf photo on the Disease page\n\n"
            "Add a valid ANTHROPIC_API_KEY to enable real Claude-powered replies."
        )
    else:
        body = (
            "I hear you. I am the AgriSense farm assistant (currently in demo mode, "
            "since the Anthropic API key on this server is not valid).\n\n"
            "Replies are demo responses, not real Claude output. Once you set a "
            "valid ANTHROPIC_API_KEY and restart Flask, I will switch to live mode.\n\n"
            "I can demo-respond to: crop recommendations, yield forecasts, weather, "
            "and disease questions."
        )
    return body + "\n\n" + DEMO_BANNER
