from flask import Blueprint, render_template, request, flash, redirect, url_for
from web.web_actions import get_past_events, predict_next_event
from operations.data.loader import load_data
import os

DATA_FILE = os.path.join('data', 'female_athletes_2425_full_stats_with_ranks.csv')

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Pagrindinis puslapis."""
    past_events = get_past_events()
    return render_template('index.html', past_events=past_events)

@web_bp.route('/show_results', methods=['POST'])
def show_results():
    """Rodo pasirinkto etapo rezultatus."""
    event = request.form.get('event')
    try:
        # Naudojame load_data vietoj tiesioginio skaitymo, kad būtų nuoseklumas
        df = load_data(DATA_FILE)
        
        # Tikriname, ar etapas egzistuoja
        if event not in df.columns:
            flash(f"Etapas '{event}' nerastas duomenyse", "error")
            return redirect(url_for('web.index'))
            
        # Filtruojame tik tas sportininkes, kurios dalyvavo etape
        results = df[['FullName', 'Nation', event]].dropna()
        # Rūšiuojame pagal rezultatą (mažesnė vertė = geresnė vieta)
        results = results.sort_values(event)
        
        # Konvertuojam į sąrašą žodynų
        results_list = []
        for _, row in results.head(10).iterrows():
            rank_value = row[event]
            # Užtikriname, kad rangas yra sveikasis skaičius
            try:
                rank = int(rank_value)
            except (ValueError, TypeError):
                rank = float(rank_value)
                
            results_list.append({
                "name": row['FullName'],
                "nation": row['Nation'],
                "rank": rank
            })
        
        return render_template('result.html', results=results_list, event=event, is_past=True)
    except Exception as e:
        flash(f"Klaida rodant rezultatus: {str(e)}", "error")
        return redirect(url_for('web.index'))
    
@web_bp.route('/predict', methods=['POST'])
def predict():
    """Prognozuoja būsimo etapo rezultatus."""
    race_type = request.form.get('race_type')
    predictions = predict_next_event(race_type)
    
    return render_template('result.html', 
                          results=predictions, 
                          event=f"Būsimas {race_type} etapas", 
                          is_past=False)

