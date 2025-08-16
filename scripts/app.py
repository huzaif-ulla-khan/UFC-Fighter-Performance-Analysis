import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UFC Fighter Analytics Dashboard",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #ff4b4b;
    }
    .fighter-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    .vs-indicator {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #ff4b4b;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare UFC fight data"""
    try:
        # Load from CSV
        df = pd.read_csv('data/ufc_fights.csv')
        st.success("✅ Successfully loaded UFC fights data from CSV!")
        
        # Data preprocessing and validation
        if 'date' in df.columns:
            # Handle different date formats flexibly
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = df['date'].isna().sum()
            if invalid_dates > 0:
                st.warning(f"⚠️ Found {invalid_dates} rows with invalid dates. Removing them.")
                df = df.dropna(subset=['date'])
        
        # Validate required columns
        required_columns = ['winner', 'loser', 'method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.error("Please ensure your CSV has columns: winner, loser, method, date")
            return None
            
        # Clean up any NaN values in critical columns
        df = df.dropna(subset=['winner', 'loser'])
        
        # Normalize method column for better KO detection
        if 'method' in df.columns:
            df['method'] = df['method'].str.strip()
            # Standardize KO/TKO variations
            df['method'] = df['method'].replace({
                'KO': 'KO/TKO',
                'TKO': 'KO/TKO',
                'Knockout': 'KO/TKO',
                'Technical Knockout': 'KO/TKO',
                'KO/TKO ': 'KO/TKO',  # Remove trailing spaces
            })
        
        st.info(f"📈 Loaded {len(df)} fight records")
        if 'date' in df.columns:
            st.info(f"📅 Date range: {df['date'].dt.year.min()} to {df['date'].dt.year.max()}")
        
        # Display unique methods for debugging
        if 'method' in df.columns:
            unique_methods = df['method'].unique()
            st.sidebar.write("**Methods in dataset:**")
            for method in sorted(unique_methods):
                count = len(df[df['method'] == method])
                st.sidebar.write(f"- {method}: {count}")
        
        return df
        
    except FileNotFoundError:
        st.error("❌ CSV file 'data/ufc_fights.csv' not found!")
        st.error("Please ensure you have:")
        st.error("1. Created a 'data' folder in your project directory")
        st.error("2. Placed your UFC fights CSV file in the data folder")
        st.error("3. Named the file 'ufc_fights.csv'")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return None

def calculate_fighter_stats(df, fighter_name):
    """Calculate comprehensive stats for a fighter"""
    fighter_wins = df[df['winner'] == fighter_name].copy()
    fighter_losses = df[df['loser'] == fighter_name].copy()
    
    if len(fighter_wins) == 0 and len(fighter_losses) == 0:
        st.warning(f"No fights found for {fighter_name}")
        return None
    
    # Add result column for easier processing
    fighter_wins['result'] = 'Win'
    fighter_losses['result'] = 'Loss'
    
    # For wins, the fighter's stats are in the winner columns
    # For losses, the fighter's stats are in the loser columns
    wins_data = fighter_wins.copy()
    losses_data = fighter_losses.copy()
    
    # Rename columns to standardize
    if 'winner_strikes_landed' in wins_data.columns:
        wins_data = wins_data.rename(columns={
            'winner_strikes_landed': 'strikes_landed',
            'winner_strikes_attempted': 'strikes_attempted',
            'winner_takedowns': 'takedowns_landed',
            'winner_takedown_attempts': 'takedown_attempts'
        })
    else:
        # If detailed stats columns don't exist, create dummy ones
        wins_data['strikes_landed'] = 0
        wins_data['strikes_attempted'] = 0
        wins_data['takedowns_landed'] = 0
        wins_data['takedown_attempts'] = 0
    
    if 'loser_strikes_landed' in losses_data.columns:
        losses_data = losses_data.rename(columns={
            'loser_strikes_landed': 'strikes_landed',
            'loser_strikes_attempted': 'strikes_attempted',
            'loser_takedowns': 'takedowns_landed',
            'loser_takedown_attempts': 'takedown_attempts'
        })
    else:
        # If detailed stats columns don't exist, create dummy ones
        losses_data['strikes_landed'] = 0
        losses_data['strikes_attempted'] = 0
        losses_data['takedowns_landed'] = 0
        losses_data['takedown_attempts'] = 0
    
    # Combine all fights
    all_fights = pd.concat([wins_data, losses_data], ignore_index=True)
    if 'date' in all_fights.columns:
        all_fights = all_fights.sort_values('date')
    
    # Calculate basic stats
    total_fights = len(all_fights)
    wins = len(fighter_wins)
    losses = len(fighter_losses)
    
    # Calculate finish rates - Fixed KO detection
    ko_wins = 0
    sub_wins = 0
    
    if 'method' in fighter_wins.columns:
        # More flexible KO detection
        ko_methods = fighter_wins['method'].str.contains('KO|TKO|Knockout|knockout', case=False, na=False)
        ko_wins = ko_methods.sum()
        
        # Submission detection
        sub_methods = fighter_wins['method'].str.contains('Sub|submission|Submission|choke|arm|leg|triangle|rear naked|guillotine', case=False, na=False)
        sub_wins = sub_methods.sum()
        
        # Debug info
        st.sidebar.write(f"**{fighter_name} Win Methods:**")
        for method in fighter_wins['method'].value_counts().index:
            count = fighter_wins['method'].value_counts()[method]
            st.sidebar.write(f"- {method}: {count}")
    
    # Striking accuracy
    total_strikes_landed = all_fights['strikes_landed'].sum()
    total_strikes_attempted = all_fights['strikes_attempted'].sum()
    strike_accuracy = (total_strikes_landed / total_strikes_attempted * 100) if total_strikes_attempted > 0 else 0
    
    # Takedown accuracy
    total_takedowns_landed = all_fights['takedowns_landed'].sum()
    total_takedown_attempts = all_fights['takedown_attempts'].sum()
    takedown_accuracy = (total_takedowns_landed / total_takedown_attempts * 100) if total_takedown_attempts > 0 else 0
    
    # Average fight time
    if 'time_minutes' in all_fights.columns:
        avg_fight_time = all_fights['time_minutes'].mean()
    else:
        avg_fight_time = 0
    
    stats = {
        'name': fighter_name,
        'total_fights': total_fights,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / total_fights * 100) if total_fights > 0 else 0,
        'ko_rate': (ko_wins / wins * 100) if wins > 0 else 0,
        'sub_rate': (sub_wins / wins * 100) if wins > 0 else 0,
        'strike_accuracy': strike_accuracy,
        'takedown_accuracy': takedown_accuracy,
        'avg_fight_time': avg_fight_time,
        'ko_wins': ko_wins,
        'sub_wins': sub_wins,
        'fights_data': all_fights
    }
    
    return stats

def create_win_loss_timeline(fighter_stats):
    """Create win/loss timeline chart"""
    if not fighter_stats or fighter_stats['fights_data'].empty:
        return go.Figure()
    
    fights = fighter_stats['fights_data'].copy()
    fights['cumulative_wins'] = (fights['result'] == 'Win').cumsum()
    fights['cumulative_losses'] = (fights['result'] == 'Loss').cumsum()
    
    fig = go.Figure()
    
    if 'date' in fights.columns:
        x_data = fights['date']
        x_title = "Date"
    else:
        x_data = range(len(fights))
        x_title = "Fight Number"
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=fights['cumulative_wins'],
        mode='lines+markers',
        name='Wins',
        line=dict(color='green', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=fights['cumulative_losses'],
        mode='lines+markers',
        name='Losses',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"{fighter_stats['name']} - Career Win/Loss Timeline",
        xaxis_title=x_title,
        yaxis_title="Cumulative Count",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_performance_radar(fighter1_stats, fighter2_stats):
    """Create radar chart comparing two fighters"""
    if not fighter1_stats or not fighter2_stats:
        return go.Figure()
    
    categories = ['Win Rate', 'Strike Accuracy', 'Takedown Accuracy', 'KO Rate', 'Submission Rate']
    
    fighter1_values = [
        fighter1_stats['win_rate'],
        fighter1_stats['strike_accuracy'],
        fighter1_stats['takedown_accuracy'],
        fighter1_stats['ko_rate'],
        fighter1_stats['sub_rate']
    ]
    
    fighter2_values = [
        fighter2_stats['win_rate'],
        fighter2_stats['strike_accuracy'],
        fighter2_stats['takedown_accuracy'],
        fighter2_stats['ko_rate'],
        fighter2_stats['sub_rate']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=fighter1_values,
        theta=categories,
        fill='toself',
        name=fighter1_stats['name'],
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=fighter2_values,
        theta=categories,
        fill='toself',
        name=fighter2_stats['name'],
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Fighter Performance Comparison",
        height=500
    )
    
    return fig

def create_finish_method_pie(fighter_stats):
    """Create pie chart showing finish methods"""
    if not fighter_stats:
        return go.Figure()
    
    wins = fighter_stats['wins']
    ko_wins = fighter_stats['ko_wins']
    sub_wins = fighter_stats['sub_wins']
    decision_wins = wins - ko_wins - sub_wins
    
    if wins == 0:
        st.warning(f"No wins found for {fighter_stats['name']}")
        return go.Figure()
    
    labels = []
    values = []
    colors = []
    
    if ko_wins > 0:
        labels.append('KO/TKO')
        values.append(ko_wins)
        colors.append('#ff4444')
    
    if sub_wins > 0:
        labels.append('Submission')
        values.append(sub_wins)
        colors.append('#44ff44')
    
    if decision_wins > 0:
        labels.append('Decision')
        values.append(decision_wins)
        colors.append('#4444ff')
    
    if not labels:
        return go.Figure()
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.3
    )])
    
    fig.update_layout(
        title=f"{fighter_stats['name']} - Win Methods Distribution",
        height=400
    )
    
    return fig

def validate_data_format(df):
    """Validate and display data format information"""
    st.sidebar.header("📊 Data Info")
    
    with st.sidebar.expander("Dataset Overview"):
        st.write(f"**Total Records**: {len(df)}")
        if 'date' in df.columns:
            st.write(f"**Date Range**: {df['date'].dt.year.min()} - {df['date'].dt.year.max()}")
        st.write(f"**Unique Fighters**: {len(set(df['winner'].tolist() + df['loser'].tolist()))}")
        if 'weight_class' in df.columns:
            st.write(f"**Weight Classes**: {df['weight_class'].nunique()}")
        
        # Show column names
        st.write("**Available Columns**:")
        for col in df.columns:
            st.write(f"- {col}")
        
        # Show sample data
        st.write("**Sample Records**:")
        display_cols = ['winner', 'loser', 'method']
        if 'date' in df.columns:
            display_cols.insert(0, 'date')
        st.dataframe(df[display_cols].head(3), hide_index=True)
    
    return True

# Main application
def main():
    st.title("🥊 UFC Fighter Performance Analytics Dashboard")
    
    # Load data
    df = load_data()
    
    if df is None or len(df) == 0:
        st.error("❌ No data available. Please check your CSV file.")
        st.info("**Required CSV format:**")
        st.info("- Columns: winner, loser, method, date (optional)")
        st.info("- Additional optional columns: weight_class, round, time_minutes, striking stats, etc.")
        return
    
    # Validate data format
    validate_data_format(df)
    
    # Get unique fighters
    all_fighters = sorted(list(set(df['winner'].tolist() + df['loser'].tolist())))
    
    # Sidebar controls
    st.sidebar.header("Fighter Selection")
    fighter1 = st.sidebar.selectbox("Select Fighter 1:", all_fighters, index=0)
    fighter2 = st.sidebar.selectbox("Select Fighter 2:", all_fighters, index=1 if len(all_fighters) > 1 else 0)
    
    if fighter1 == fighter2:
        st.sidebar.warning("Please select different fighters for comparison")
        return
    
    # Calculate fighter stats
    fighter1_stats = calculate_fighter_stats(df, fighter1)
    fighter2_stats = calculate_fighter_stats(df, fighter2)
    
    if not fighter1_stats or not fighter2_stats:
        st.error("Unable to calculate stats for selected fighters")
        return
    
    # Main dashboard layout
    col1, col2 = st.columns(2)
    
    # Fighter 1 Card
    with col1:
        st.markdown(f"""
        <div class="fighter-card">
            <h2>{fighter1_stats['name']}</h2>
            <h3>Record: {fighter1_stats['wins']}-{fighter1_stats['losses']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.metric("Win Rate", f"{fighter1_stats['win_rate']:.1f}%")
            st.metric("Strike Accuracy", f"{fighter1_stats['strike_accuracy']:.1f}%")
        with col1_2:
            st.metric("KO Rate", f"{fighter1_stats['ko_rate']:.1f}%")
            st.metric("Takedown Accuracy", f"{fighter1_stats['takedown_accuracy']:.1f}%")
    
    # Fighter 2 Card
    with col2:
        st.markdown(f"""
        <div class="fighter-card">
            <h2>{fighter2_stats['name']}</h2>
            <h3>Record: {fighter2_stats['wins']}-{fighter2_stats['losses']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Win Rate", f"{fighter2_stats['win_rate']:.1f}%")
            st.metric("Strike Accuracy", f"{fighter2_stats['strike_accuracy']:.1f}%")
        with col2_2:
            st.metric("KO Rate", f"{fighter2_stats['ko_rate']:.1f}%")
            st.metric("Takedown Accuracy", f"{fighter2_stats['takedown_accuracy']:.1f}%")
    
    # VS indicator
    st.markdown('<div class="vs-indicator">VS</div>', unsafe_allow_html=True)
    
    # Charts section
    st.header("Performance Analytics")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Comparison", "Fight Methods", "Detailed Stats"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_win_loss_timeline(fighter1_stats)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = create_win_loss_timeline(fighter2_stats)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        fig_radar = create_performance_radar(fighter1_stats, fighter2_stats)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Head-to-head comparison table
        st.subheader("Head-to-Head Statistics")
        comparison_data = {
            'Metric': ['Total Fights', 'Wins', 'Losses', 'Win Rate (%)', 'KO/TKO Wins', 'Submission Wins', 
                      'Strike Accuracy (%)', 'Takedown Accuracy (%)', 'Avg Fight Time (min)'],
            fighter1: [
                fighter1_stats['total_fights'], fighter1_stats['wins'], fighter1_stats['losses'],
                f"{fighter1_stats['win_rate']:.1f}", fighter1_stats['ko_wins'], fighter1_stats['sub_wins'],
                f"{fighter1_stats['strike_accuracy']:.1f}", f"{fighter1_stats['takedown_accuracy']:.1f}",
                f"{fighter1_stats['avg_fight_time']:.1f}"
            ],
            fighter2: [
                fighter2_stats['total_fights'], fighter2_stats['wins'], fighter2_stats['losses'],
                f"{fighter2_stats['win_rate']:.1f}", fighter2_stats['ko_wins'], fighter2_stats['sub_wins'],
                f"{fighter2_stats['strike_accuracy']:.1f}", f"{fighter2_stats['takedown_accuracy']:.1f}",
                f"{fighter2_stats['avg_fight_time']:.1f}"
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie1 = create_finish_method_pie(fighter1_stats)
            st.plotly_chart(fig_pie1, use_container_width=True)
        with col2:
            fig_pie2 = create_finish_method_pie(fighter2_stats)
            st.plotly_chart(fig_pie2, use_container_width=True)
    
    with tab4:
        st.subheader(f"{fighter1} - Recent Fights")
        if not fighter1_stats['fights_data'].empty:
            display_cols = ['result', 'method']
            if 'date' in fighter1_stats['fights_data'].columns:
                display_cols.insert(0, 'date')
            if 'round' in fighter1_stats['fights_data'].columns:
                display_cols.append('round')
            if 'time_minutes' in fighter1_stats['fights_data'].columns:
                display_cols.append('time_minutes')
            
            recent_fights_f1 = fighter1_stats['fights_data'][display_cols].head(10)
            st.dataframe(recent_fights_f1, use_container_width=True)
        
        st.subheader(f"{fighter2} - Recent Fights")
        if not fighter2_stats['fights_data'].empty:
            display_cols = ['result', 'method']
            if 'date' in fighter2_stats['fights_data'].columns:
                display_cols.insert(0, 'date')
            if 'round' in fighter2_stats['fights_data'].columns:
                display_cols.append('round')
            if 'time_minutes' in fighter2_stats['fights_data'].columns:
                display_cols.append('time_minutes')
            
            recent_fights_f2 = fighter2_stats['fights_data'][display_cols].head(10)
            st.dataframe(recent_fights_f2, use_container_width=True)
    
    # Additional insights
    st.header("Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if fighter1_stats['win_rate'] > fighter2_stats['win_rate']:
            st.success(f"🏆 {fighter1} has a higher win rate ({fighter1_stats['win_rate']:.1f}% vs {fighter2_stats['win_rate']:.1f}%)")
        else:
            st.success(f"🏆 {fighter2} has a higher win rate ({fighter2_stats['win_rate']:.1f}% vs {fighter1_stats['win_rate']:.1f}%)")
    
    with col2:
        if fighter1_stats['strike_accuracy'] > fighter2_stats['strike_accuracy']:
            st.info(f"🎯 {fighter1} has better striking accuracy ({fighter1_stats['strike_accuracy']:.1f}% vs {fighter2_stats['strike_accuracy']:.1f}%)")
        else:
            st.info(f"🎯 {fighter2} has better striking accuracy ({fighter2_stats['strike_accuracy']:.1f}% vs {fighter1_stats['strike_accuracy']:.1f}%)")
    
    with col3:
        if fighter1_stats['ko_rate'] > fighter2_stats['ko_rate']:
            st.warning(f"💥 {fighter1} has a higher knockout rate ({fighter1_stats['ko_rate']:.1f}% vs {fighter2_stats['ko_rate']:.1f}%)")
        else:
            st.warning(f"💥 {fighter2} has a higher knockout rate ({fighter2_stats['ko_rate']:.1f}% vs {fighter1_stats['ko_rate']:.1f}%)")


    # Footer Section
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; margin-top: 50px;'>
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; 
                    border-radius: 15px; 
                    color: white; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='margin: 0 0 10px 0; font-weight: 300;'>🥊 UFC Fighter Analytics Dashboard</h3>
            <p style='margin: 0 0 15px 0; opacity: 0.9; font-size: 14px;'>
                Advanced analytics and insights for UFC fighters and enthusiasts
            </p>
            <div style='border-top: 1px solid rgba(255,255,255,0.2); 
                        padding-top: 15px; 
                        margin-top: 15px;'>
                <p style='margin: 0; font-size: 16px; font-weight: 500;'>
                    💜 Developed with passion by <strong>Huzaif Ulla Khan</strong>
                </p>
                <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>
                    Built with Streamlit • Plotly • Python
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()