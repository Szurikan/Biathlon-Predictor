from flask import Blueprint, render_template, request, flash, redirect, url_for
from web.web_actions import get_past_events, predict_next_event
from models.RandomForest.predict_place import predict_place_with_participation
from models.RandomForest.predict_participation import predict_participation
from operations.data.loader import load_data
import pandas as pd
import os
from operations.predict_all_events import train_model



# CLEANED_CSV = "data/female_athletes_cleaned_final.csv"
# BINARY_CSV = "data/female_athletes_binary_competitions.csv"
DATA_FILE = os.path.join('data', 'female_athletes_2425_full_stats_with_ranks.csv')
DB_FILE = "data/athletes_data.db"

web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def index():
    past_events = get_past_events()
    return render_template('index.html', past_events=past_events)

@web_bp.route('/show_results', methods=['POST'])
def show_results():
    """Rodo pasirinkto etapo rezultatus."""
    event = request.form.get('event')
    try:
        # Naudojame load_data 
        df = load_data(DB_FILE)
        
        # Tikriname, ar etapas egzistuoja
        if event not in df.columns:
            flash(f"Etapas '{event}' nerastas duomenyse", "error")
            return redirect(url_for('web.index'))
            
        # Filtruojame tik tas sportininkes, kurios dalyvavo etape
        results = df[['FullName', event]].dropna()
        # Rusiuojame pagal rezultata
        results = results.sort_values(event)
        
        # konvertuojame i sarasa zodynu
        results_list = []
        for _, row in results.head(5).iterrows():
            rank_value = row[event]
            # darom kad butu sveikasis skaicius
            try:
                rank = int(rank_value)
            except (ValueError, TypeError):
                rank = float(rank_value)
                
            results_list.append({
                "name": row['FullName'],
                "rank": rank
            })
        
        return render_template('result.html', results=results_list, event=event, is_past=True)
    except Exception as e:
        flash(f"Klaida rodant rezultatus: {str(e)}", "error")
        return redirect(url_for('web.index'))
    
@web_bp.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model')
    event = request.form.get('event')

    if model_name != "random_forest":
        flash("Šiuo metu palaikomas tik 'Random Forest' modelis.", "error")
        return redirect(url_for("web.index"))

    try:
        # 1. Paleidžiam dalyvavimo prognozę
        predict_participation(
            data_path=DB_FILE,
            target_column=event,
            output_dir="data"
        )

        predict_place_with_participation(
            data_path=DB_FILE,
            target_column=event,
            output_dir="models/RandomForest"
        )

        # 3. Nuskaitom rezultatų CSV
        filename = f"predicted_places_{event.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        filepath = os.path.join("models/RandomForest", filename)
        df = pd.read_csv(filepath)

        results_list = df[["PredictedRank", "FullName", "PredictedPlace", "ActualPlace"]].to_dict(orient="records")

        return render_template("result.html", results=results_list, event=event, is_past=False)
    
    except Exception as e:
        flash(f"Klaida prognozuojant: {str(e)}", "error")
        return redirect(url_for("web.index"))
    
@web_bp.route('/predict_next', methods=['POST'])
def predict_next():
    model_name = request.form.get('model')
    event_type = request.form.get('event_type')

    if model_name not in ("random_forest", "xgboost", "lstm"):
        flash(f"Modelis '{model_name}' nepalaikomas.", "error")
        return redirect(url_for('web.index'))

    try:
        results = predict_next_event(event_type, model_name)

        return render_template(
            'next_result.html',
            results=results,
            model=model_name,
            event_type=event_type
        )
    except Exception as e:
        flash(f"Klaida prognozuojant: {e}", "error")
        return redirect(url_for('web.index'))

@web_bp.route('/train_models', methods=['POST'])
def train_models():
    model = request.form.get("model")  # 'random_forest', 'xgboost', 'lstm'
    part = request.form.get("task")    # 'participation', 'place', 'both'

    model_map = {
        "random_forest": "RandomForest",
        "xgboost": "XGBoost",
        "lstm": "LSTM"
    }

    if model not in model_map or part not in ["participation", "place", "both"]:
        flash("Neteisingi pasirinkimai.", "error")
        return redirect(url_for('web.index'))

    try:
        train_model(model_map[model], part)
        flash("Modeliai sėkmingai apmokyti!", "success")
    except Exception as e:
        flash(f"Klaida treniruojant modelius: {str(e)}", "error")

    return redirect(url_for("web.index"))


