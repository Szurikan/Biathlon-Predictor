from flask import Blueprint, render_template
from web.web_actions import get_past_events

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Pagrindinis puslapis."""
    past_events = get_past_events()
    return render_template('index.html', past_events=past_events)