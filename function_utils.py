# Functions

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.patches import Ellipse
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import zscore
import seaborn as sns
from mplsoccer import VerticalPitch
from scipy.stats import percentileofscore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D

pd.set_option('display.max_columns', None)
def per_90_calc(df, source_col, new_col):
    df[new_col] = (
        pd.to_numeric(df[source_col], errors='coerce') / 
        pd.to_numeric(df['playing_time_min'], errors='coerce')
    ) * 90
    df[new_col] = df[new_col].round(2)
    return df


def plot_two_players_radar_plotly(df, attributes, player1, player2, colors=['blue', 'red'], pos_filter=None):
    """
    Plot radar chart comparing two players using percentiles.
    Optionally filters DataFrame to players whose 'pos' contains pos_filter substring.
    Returns True if plot successful, False otherwise.
    """
    # Filter DataFrame by pos if pos_filter is given
    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    df_percentiles = df_filtered.copy()

    for col in attributes:
        col_data = pd.to_numeric(df_filtered[col], errors='coerce').dropna().astype(float).values

        def get_percentile(val):
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                return 0
            if np.isnan(val_f):
                return 0
            return percentileofscore(col_data, val_f) / 100  # scale 0 to 1

        df_percentiles[col] = df[col].apply(get_percentile)

    row1_percentile = df_percentiles[df_percentiles['player'] == player1]
    row2_percentile = df_percentiles[df_percentiles['player'] == player2]
    row1_raw = df[df['player'] == player1]
    row2_raw = df[df['player'] == player2]

    if row1_percentile.empty or row1_raw.empty:
        st.error(f"Player **{player1}** does not play in the selected position or is not found.")
        return False
    if row2_percentile.empty or row2_raw.empty:
        st.error(f"Player **{player2}** does not play in the selected position or is not found.")
        return False

    perc_values_1 = row1_percentile[attributes].values.flatten().tolist()
    perc_values_2 = row2_percentile[attributes].values.flatten().tolist()
    raw_values_1 = row1_raw[attributes].values.flatten().tolist()
    raw_values_2 = row2_raw[attributes].values.flatten().tolist()

    perc_values_1 += perc_values_1[:1]
    perc_values_2 += perc_values_2[:1]
    raw_values_1 += raw_values_1[:1]
    raw_values_2 += raw_values_2[:1]
    labels = attributes + [attributes[0]]

    hover_texts = []
    for attr, val1, val2 in zip(labels, raw_values_1, raw_values_2):
        val1_str = f"{float(val1):.1f}".rstrip('0').rstrip('.') if pd.notnull(val1) else "N/A"
        val2_str = f"{float(val2):.1f}".rstrip('0').rstrip('.') if pd.notnull(val2) else "N/A"
        hover_texts.append(
            f"{attr}:<br>"
            f"<span style='color:{colors[0]}; font-weight:bold'>{player1}</span>: {val1_str}<br>"
            f"<span style='color:{colors[1]}; font-weight:bold'>{player2}</span>: {val2_str}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=perc_values_1,
        theta=labels,
        fill='toself',
        name=player1,
        line=dict(color=colors[0]),
        hoverinfo='text',
        text=hover_texts
    ))
    fig.add_trace(go.Scatterpolar(
        r=perc_values_2,
        theta=labels,
        fill='toself',
        name=player2,
        line=dict(color=colors[1]),
        hoverinfo='text',
        text=hover_texts
    ))

    fig.update_layout(
        title=dict(
            text="Radar Chart Comparison (Percentiles)",
            x=0.45,
            xanchor='center',
            font=dict(size=18, family='Arial Black')
        ),
        polar=dict(
            domain=dict(x=[0, 1], y=[0, 1]),
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                showline=False,
                ticks='',
                range=[0, 1]
            )
        ),
        showlegend=True,
        width=1200,
        height=700,
        margin=dict(l=30, r=30, t=50, b=30),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.5)")
    )

    st.plotly_chart(fig, use_container_width=True)
    return True

def get_two_players_radar_figure(df, attributes, player1, player2=None, pos_filter=None):
    import numpy as np
    import pandas as pd
    from scipy.stats import percentileofscore
    import plotly.graph_objects as go

    # Filter DataFrame by position if given
    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    df_percentiles = df_filtered.copy()

    # Compute percentiles for each attribute
    for col in attributes:
        col_data = pd.to_numeric(df_filtered[col], errors='coerce').dropna().astype(float).values

        def get_percentile(val):
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                return 0
            if np.isnan(val_f):
                return 0
            return percentileofscore(col_data, val_f) / 100  # scale 0-1

        df_percentiles[col] = df[col].apply(get_percentile)

    def get_player_data(player_name):
        perc_row = df_percentiles[df_percentiles['player'] == player_name]
        raw_row = df[df['player'] == player_name]
        return perc_row, raw_row

    row1_percentile, row1_raw = get_player_data(player1)
    if row1_percentile.empty or row1_raw.empty:
        return None, f"Player **{player1}** does not play in the selected position or is not found."

    if player2:
        row2_percentile, row2_raw = get_player_data(player2)
        if row2_percentile.empty or row2_raw.empty:
            return None, f"Player **{player2}** does not play in the selected position or is not found."
    else:
        row2_percentile, row2_raw = None, None

    # Average percentiles across all players (for comparison)
    avg_percentiles = df_percentiles[attributes].mean().values.tolist()
    avg_percentiles += avg_percentiles[:1]  # close the radar

    perc_values_1 = row1_percentile[attributes].values.flatten().tolist()
    raw_values_1 = row1_raw[attributes].values.flatten().tolist()
    perc_values_1 += perc_values_1[:1]
    raw_values_1 += raw_values_1[:1]

    if player2:
        perc_values_2 = row2_percentile[attributes].values.flatten().tolist()
        raw_values_2 = row2_raw[attributes].values.flatten().tolist()
        perc_values_2 += perc_values_2[:1]
        raw_values_2 += raw_values_2[:1]

    labels = attributes + [attributes[0]]
    colors = ['blue', 'red']
    hover_texts = []

    for i, attr in enumerate(labels):
        val1 = raw_values_1[i] if i < len(raw_values_1) else None
        val1_str = f"{float(val1):.1f}".rstrip('0').rstrip('.') if pd.notnull(val1) else "N/A"
        hover = f"{attr}:<br><span style='color:{colors[0]}; font-weight:bold'>{player1}</span>: {val1_str}"

        if player2:
            val2 = raw_values_2[i] if i < len(raw_values_2) else None
            val2_str = f"{float(val2):.1f}".rstrip('0').rstrip('.') if pd.notnull(val2) else "N/A"
            hover += f"<br><span style='color:{colors[1]}; font-weight:bold'>{player2}</span>: {val2_str}"

        hover_texts.append(hover)

    fig = go.Figure()

    # Player 1 trace
    fig.add_trace(go.Scatterpolar(
        r=perc_values_1,
        theta=labels,
        fill='toself',
        name=player1,
        line=dict(color=colors[0]),
        hoverinfo='text',
        text=hover_texts
    ))

    # Player 2 trace (if provided)
    if player2:
        fig.add_trace(go.Scatterpolar(
            r=perc_values_2,
            theta=labels,
            fill='toself',
            name=player2,
            line=dict(color=colors[1]),
            hoverinfo='text',
            text=hover_texts
        ))

    # Average trace
    fig.add_trace(go.Scatterpolar(
        r=avg_percentiles,
        theta=labels,
        name="Average",
        line=dict(color="gray", dash="dash"),
        fill='none',
        hoverinfo='skip',
        showlegend=True
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                showline=False,
                ticks='',
                range=[0, 1]
            )
        ),
        showlegend=True,
        width=1200,
        height=700,
        margin=dict(l=30, r=30, t=50, b=30),
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    return fig, None


def format_value(val):
    """Format value to 1 decimal place, removing trailing .0"""
    if pd.isna(val):
        return ''
    val_rounded = round(val, 1)
    if val_rounded.is_integer():
        return str(int(val_rounded))
    return str(val_rounded)

def get_two_players_comparison_matrix(df, attributes, player1, player2, pos_filter=None):
    """
    Returns a styled matrix comparing two players:
    - Shows raw values and percentiles
    - Heatmap coloring applied to percentile columns
    - Values formatted cleanly
    - Optionally filters DataFrame by 'pos' column containing pos_filter substring
    """
    players_data = []

    # Filter df by pos if pos_filter given
    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, case=False, na=False)]
    else:
        df_filtered = df.copy()

    for player in [player1, player2]:
        player_row = df[df['player'] == player]  # Always pick from original df for raw values
        if player_row.empty:
            print(f"Player ID '{player}' not found in DataFrame.")
            continue

        player_name = player_row['player'].values[0]
        col_data = {}

        for col in attributes:
            full_col = pd.to_numeric(df_filtered[col], errors='coerce').dropna()  # percentile base from filtered df
            val = pd.to_numeric(player_row[col].values[0], errors='coerce')

            if pd.isna(val) or full_col.empty:
                raw_val = np.nan
                percentile = np.nan
            else:
                raw_val = val
                percentile = percentileofscore(full_col, val)

            col_data[col] = [format_value(raw_val), format_value(percentile)]

        player_df = pd.DataFrame(col_data, index=["Value", "Percentile"]).T
        player_df.columns = pd.MultiIndex.from_product([[player_name], player_df.columns])
        players_data.append(player_df)

    # Combine both players into one DataFrame
    matrix = pd.concat(players_data, axis=1)

    # Apply color styling to "Percentile" columns only
    def highlight_percentiles(val):
        try:
            val = float(val)
        except:
            return ''
        color = f'background-color: rgba({255 - int(val*2.55)}, {int(val*2.55)}, 100, 0.6);'
        return color

    percent_mask = matrix.columns.get_level_values(1) == "Percentile"
    styled = matrix.style.applymap(highlight_percentiles, subset=pd.IndexSlice[:, percent_mask])
    return styled


def plot_two_players_histograms(df, attributes, player1, player2, colors=['blue', 'red'], pos_filter=None, bins=20):
    """
    Plot multiple histograms (one per attribute) showing distribution among players.
    Highlight two selected players using vertical lines.
    Optionally filter by player position using pos_filter.
    """
    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    player1_row = df[df['player'] == player1]
    player2_row = df[df['player'] == player2]

    if player1_row.empty:
        print(f"Player '{player1}' not found in DataFrame.")
        return
    if player2_row.empty:
        print(f"Player '{player2}' not found in DataFrame.")
        return

    num_attrs = len(attributes)
    fig = make_subplots(
        rows=(num_attrs + 1) // 2,
        cols=2,
        subplot_titles=attributes,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    row, col = 1, 1
    for attr in attributes:
        data = pd.to_numeric(df_filtered[attr], errors='coerce').dropna()

        try:
            player1_val = float(player1_row[attr].values[0])
        except (ValueError, TypeError):
            player1_val = np.nan

        try:
            player2_val = float(player2_row[attr].values[0])
        except (ValueError, TypeError):
            player2_val = np.nan

        # Histogram trace
        hist_trace = go.Histogram(
            x=data,
            nbinsx=bins,
            marker=dict(color='lightgrey'),
            name='All Players',
            showlegend=False
        )
        fig.add_trace(hist_trace, row=row, col=col)

        max_y = data.value_counts().max()

        # Line for player1
        if not np.isnan(player1_val):
            line1 = go.Scatter(
                x=[player1_val, player1_val],
                y=[0, max_y],
                mode='lines',
                line=dict(color=colors[0], width=3),
                name=player1 if (row == 1 and col == 1) else None,
                showlegend=(row == 1 and col == 1),
                hovertemplate=f"<b>{player1}</b>: {player1_val:.2f}<extra></extra>"
            )
            fig.add_trace(line1, row=row, col=col)

        # Line for player2
        if not np.isnan(player2_val):
            line2 = go.Scatter(
                x=[player2_val, player2_val],
                y=[0, max_y],
                mode='lines',
                line=dict(color=colors[1], width=3),
                name=player2 if (row == 1 and col == 1) else None,
                showlegend=(row == 1 and col == 1),
                hovertemplate=f"<b>{player2}</b>: {player2_val:.2f}<extra></extra>"
            )
            fig.add_trace(line2, row=row, col=col)

        # Move to next subplot position
        if col == 1:
            col = 2
        else:
            col = 1
            row += 1

    fig.update_layout(
        title_text="Attribute Distribution Comparison with Highlighted Players",
        height=400 * ((num_attrs + 1) // 2),
        width=1000,
        bargap=0.1,
        showlegend=True,
        template="plotly_white",
        margin=dict(t=80, l=50, r=50, b=50)
    )

    fig.show()


def plot_two_players_histogram_plots_matplotlib(df, attributes, player1, player2=None, colors=['blue', 'red'], pos_filter=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    player1_row = df[df['player'] == player1]
    player2_row = df[df['player'] == player2] if player2 else pd.DataFrame()

    if player1_row.empty:
        print(f"Player '{player1}' not found in DataFrame.")
        return
    if player2 and player2_row.empty:
        print(f"Player '{player2}' not found in DataFrame.")
        return

    num_attrs = len(attributes)
    ncols = 2
    nrows = math.ceil(num_attrs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    fig.patch.set_facecolor('#262730')  # Dark background

    axes = axes.flatten()

    for i, attr in enumerate(attributes):
        ax = axes[i]
        data = pd.to_numeric(df_filtered[attr], errors='coerce').dropna()

        lower, upper = np.percentile(data, 1), np.percentile(data, 99)
        clipped_data = data[(data >= lower) & (data <= upper)]

        ax.set_facecolor('#262730')  # subplot background

        # Plot histogram
        ax.hist(clipped_data, bins=30, color='#b0b0b0', edgecolor='white', alpha=0.6, label='Population')

        mean_val = clipped_data.mean()
        avg_line = ax.axvline(mean_val, color='gray', linestyle='--', linewidth=1.5, label='Average')

        handles = [avg_line]

        try:
            player1_val = float(player1_row[attr].values[0])
            l1 = ax.axvline(player1_val, color=colors[0], linewidth=1.5, label=player1)
            handles.append(l1)
        except (ValueError, TypeError):
            pass

        if player2 and not player2_row.empty:
            try:
                player2_val = float(player2_row[attr].values[0])
                l2 = ax.axvline(player2_val, color=colors[1], linewidth=1.5, label=player2)
                handles.append(l2)
            except (ValueError, TypeError):
                pass

        ax.set_title(attr, color='white')
        ax.set_xlabel("Value", color='white')
        ax.set_ylabel("Count", color='white')

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        ax.legend(handles=handles, loc='upper right', fontsize=9, facecolor='#262730', edgecolor='white', labelcolor='white')
        for text in ax.get_legend().get_texts():
            text.set_color('white')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_two_players_density_plots_matplotlib(df, attributes, player1, player2=None, colors=['blue', 'red'], pos_filter=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    player1_row = df[df['player'] == player1]
    player2_row = df[df['player'] == player2] if player2 else pd.DataFrame()

    if player1_row.empty:
        print(f"Player '{player1}' not found in DataFrame.")
        return
    if player2 and player2_row.empty:
        print(f"Player '{player2}' not found in DataFrame.")
        return

    num_attrs = len(attributes)
    ncols = 2
    nrows = math.ceil(num_attrs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    
    # Set entire figure background color to #262730 (dark)
    fig.patch.set_facecolor('#262730')
    
    axes = axes.flatten()

    for i, attr in enumerate(attributes):
        ax = axes[i]
        data = pd.to_numeric(df_filtered[attr], errors='coerce').dropna()

        lower, upper = np.percentile(data, 1), np.percentile(data, 99)
        clipped_data = data[(data >= lower) & (data <= upper)]

        # Set subplot background color
        ax.set_facecolor('#262730')

        # KDE plot with lighter gray fill
        sns.kdeplot(clipped_data, ax=ax, fill=True, color='#b0b0b0', linewidth=1.5)

        mean_val = clipped_data.mean()
        avg_line = ax.axvline(mean_val, color='gray', linestyle='--', linewidth=1.5, label='Average')

        handles = [avg_line]

        try:
            player1_val = float(player1_row[attr].values[0])
            l1 = ax.axvline(player1_val, color=colors[0], linewidth=1.5, label=player1)
            handles.append(l1)
        except (ValueError, TypeError):
            pass

        if player2 and not player2_row.empty:
            try:
                player2_val = float(player2_row[attr].values[0])
                l2 = ax.axvline(player2_val, color=colors[1], linewidth=1.5, label=player2)
                handles.append(l2)
            except (ValueError, TypeError):
                pass

        ax.set_title(attr, color='white')
        ax.set_xlabel("Value", color='white')
        ax.set_ylabel("Density", color='white')

        # Set tick params and spines to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        ax.legend(handles=handles, loc='upper right', fontsize=9, facecolor='#262730', edgecolor='white', labelcolor='white')
        # legend text color fix below:
        for text in ax.get_legend().get_texts():
            text.set_color('white')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_conceded_vs_xg_with_bounds(data: pd.DataFrame, team: str, std_multiplier=1):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9,7))

    X = data['Expected Goals Conceded'].values.reshape(-1, 1)
    y = data['Goals Conceded'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    residuals = y - y_pred
    resid_std = np.std(residuals)

    upper_bound = y_pred + std_multiplier * resid_std
    lower_bound = y_pred - std_multiplier * resid_std

    ax.plot(data['Expected Goals Conceded'], upper_bound, color='green', linestyle='--', linewidth=0.5)
    ax.plot(data['Expected Goals Conceded'], lower_bound, color='green', linestyle='--', linewidth=0.5)

    overperform = residuals < -std_multiplier * resid_std
    underperform = residuals > std_multiplier * resid_std
    normal = ~(overperform | underperform)

    ax.scatter(data.loc[normal, 'Expected Goals Conceded'], data.loc[normal, 'Goals Conceded'], color='grey', alpha=0.7, label='As expected')
    ax.scatter(data.loc[overperform, 'Expected Goals Conceded'], data.loc[overperform, 'Goals Conceded'], color='green', alpha=0.8, label='Overperforming xg')
    ax.scatter(data.loc[underperform, 'Expected Goals Conceded'], data.loc[underperform, 'Goals Conceded'], color='orange', alpha=0.8, label='Underperforming xg')

    # Annotate all teams with their names near their points
    for _, row in data.iterrows():
        ax.text(row['Expected Goals Conceded'] + 0.02, row['Goals Conceded'], row['squad'], fontsize=8, alpha=0.7)

    # Highlight selected team with bigger red star and bold red label
    team_data = data[data['squad'] == team]
    if not team_data.empty:
        ax.scatter(
            team_data['Expected Goals Conceded'],
            team_data['Goals Conceded'],
            color='red',
            marker='*',
            s=250,
            label=team
        )

    ax.set_xlabel('Expected Goals Conceded (xGC)')
    ax.set_ylabel('Goals Conceded')
    ax.set_title('Goals Conceded vs Expected xGC with Regression and Residual Bounds')
    ax.legend()
    fig.tight_layout()

    return fig



def plot_performance_vs_xg_with_bounds(data: pd.DataFrame, team: str, std_multiplier=1):
    """
    Scatter plot of Goals Scored vs Expected Goals with regression line,
    confidence bounds (upper/lower) based on residuals,
    and highlight selected team and outliers outside bounds.
    
    Parameters:
    - data: DataFrame with 'squad', 'Goals Scored', 'Expected Goals'
    - team: Team name to highlight
    - std_multiplier: number of std deviations for bounds (default=2)
    """
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))  # create figure and axes
    
    # Prepare X and y
    X = data['Expected Goals'].values.reshape(-1, 1)
    y = data['Goals Scored'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate residuals and std dev of residuals
    residuals = y - y_pred
    resid_std = np.std(residuals)
    
    # Upper and lower bounds
    upper_bound = y_pred + std_multiplier * resid_std
    lower_bound = y_pred - std_multiplier * resid_std
    
    # Plot bounds with thinner linewidth
    ax.plot(data['Expected Goals'], upper_bound, color='green', linestyle='--', linewidth=0.5)
    ax.plot(data['Expected Goals'], lower_bound, color='green', linestyle='--', linewidth=0.5)
    
    # Scatter all points, color by whether they are outside bounds
    overperform = residuals > std_multiplier * resid_std
    underperform = residuals < -std_multiplier * resid_std
    normal = ~(overperform | underperform)
    
    ax.scatter(data.loc[normal, 'Expected Goals'], data.loc[normal, 'Goals Scored'], color='grey', alpha=0.7, label='Matching xG')
    ax.scatter(data.loc[overperform, 'Expected Goals'], data.loc[overperform, 'Goals Scored'], color='green', alpha=0.8, label='Extremely efficient teams')
    ax.scatter(data.loc[underperform, 'Expected Goals'], data.loc[underperform, 'Goals Scored'], color='orange', alpha=0.8, label='Wasteful teams')
    
    # Annotate all teams with their names near their points
    for _, row in data.iterrows():
        ax.text(row['Expected Goals'] + 0.02, row['Goals Scored'], row['squad'], fontsize=8, alpha=0.7)
    
    # Highlight selected team with bigger red star and name with bold
    team_data = data[data['squad'] == team]
    if not team_data.empty:
        ax.scatter(
            team_data['Expected Goals'],
            team_data['Goals Scored'],
            color='red',
            marker='*',
            s=250,
            label=team
        )
    
    ax.set_xlabel('Expected xG')
    ax.set_ylabel('Goals')
    ax.set_title('Goals vs Expected xG with Regression and Residual Bounds')
    ax.legend()
    fig.tight_layout()
    
    return fig



def plot_two_players_boxplots_matplotlib(df, attributes, player1, player2=None, colors=['blue', 'red'], pos_filter=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    player1_row = df[df['player'] == player1]
    player2_row = df[df['player'] == player2] if player2 else pd.DataFrame()

    if player1_row.empty:
        print(f"Player '{player1}' not found in DataFrame.")
        return
    if player2 and player2_row.empty:
        print(f"Player '{player2}' not found in DataFrame.")
        return

    num_attrs = len(attributes)
    ncols = 2
    nrows = math.ceil(num_attrs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    fig.patch.set_facecolor('#262730')
    axes = axes.flatten()

    for i, attr in enumerate(attributes):
        ax = axes[i]
        data = pd.to_numeric(df_filtered[attr], errors='coerce').dropna()

        lower, upper = np.percentile(data, 1), np.percentile(data, 99)
        clipped_data = data[(data >= lower) & (data <= upper)]

        ax.set_facecolor('#262730')

        ax.boxplot(clipped_data, vert=True, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='#262730', color='white'),
                   medianprops=dict(color='white'),
                   whiskerprops=dict(color='white'),
                   capprops=dict(color='white'),
                   flierprops=dict(markerfacecolor='white', markeredgecolor='white'))

        x_pos = 1

        try:
            player1_val = float(player1_row[attr].values[0])
            if lower <= player1_val <= upper:
                ax.scatter(x_pos, player1_val, color=colors[0], s=100, label=player1, zorder=5)
            else:
                ax.scatter(x_pos, player1_val, color=colors[0], s=100, label=player1, zorder=5, marker='^')
        except (ValueError, TypeError):
            pass

        if player2 and not player2_row.empty:
            try:
                player2_val = float(player2_row[attr].values[0])
                if lower <= player2_val <= upper:
                    ax.scatter(x_pos, player2_val, color=colors[1], s=100, label=player2, zorder=5)
                else:
                    ax.scatter(x_pos, player2_val, color=colors[1], s=100, label=player2, zorder=5, marker='^')
            except (ValueError, TypeError):
                pass

        ax.set_title(attr, color='white')
        ax.set_xticks([])
        ax.set_ylabel("Value", color='white')

        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(handles=handles, loc='upper right', fontsize=9)
            frame = leg.get_frame()
            frame.set_facecolor('#262730')
            frame.set_edgecolor('white')
            frame.set_linewidth(1.5)  # make the border thicker so you can see it clearly

            # Force redraw of the legend to apply styles correctly
            frame.set_alpha(1)
            for text in leg.get_texts():
                text.set_color('white')


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig



def plot_player_shots(player_name, all_shots_df, player_number=1):
    """
    Plots the shot map for a specific player on a half-pitch using StatsBomb dimensions,
    with color depending on player number (Player 1: blue, Player 2: red).
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch
    from matplotlib.lines import Line2D

    # Ensure 'date' column is datetime
    all_shots_df['date'] = pd.to_datetime(all_shots_df['date'], errors='coerce')

    # Filter by date range
    start_date = pd.Timestamp('2024-08-16')
    end_date = pd.Timestamp('2025-05-27')
    date_filtered = all_shots_df[(all_shots_df['date'] >= start_date) & (all_shots_df['date'] <= end_date)]

    # Filter the data for the selected player
    player_shots = date_filtered[date_filtered['player'] == player_name].copy()

    if player_shots.empty:
        print(f"No shots found for player: {player_name} in the specified date range.")
        return

    # Convert to numeric and scale to StatsBomb dimensions
    player_shots['X'] = pd.to_numeric(player_shots['X'], errors='coerce') * 120
    player_shots['Y'] = pd.to_numeric(player_shots['Y'], errors='coerce') * 80
    player_shots['Y'] = 80 - player_shots['Y']

    # Convert xG to numeric and set minimum size for visibility
    player_shots['xG'] = pd.to_numeric(player_shots['xG'], errors='coerce').fillna(0)
    min_size = 50
    sizes = np.clip(player_shots['xG'] * 1000, min_size, None)

    # Drop any rows with NaNs after conversion
    player_shots.dropna(subset=['X', 'Y'], inplace=True)

    # Create half-pitch
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, pitch_color='#262730')
    fig, ax = pitch.draw(figsize=(10, 10))

    # Separate goal and non-goal shots
    goal_shots = player_shots[player_shots['result'] == 'Goal']
    non_goal_shots = player_shots[player_shots['result'] != 'Goal']

    goal_sizes = sizes[goal_shots.index]
    non_goal_sizes = sizes[non_goal_shots.index]

    # Set colors by player number
    if player_number == 1:
        goal_facecolor = (0, 0, 1, 0.2)  # Blue with alpha
        goal_edgecolor = 'blue'
    else:
        goal_facecolor = (1, 0, 0, 0.2)  # Red with alpha
        goal_edgecolor = 'red'

    # Plot non-goal shots (black)
    pitch.scatter(
        non_goal_shots['X'], 
        non_goal_shots['Y'], 
        s=non_goal_sizes,
        facecolors=(0, 0, 0, 0.2),
        edgecolors='black',
        linewidth=2,
        ax=ax,
        zorder=2
    )

    # Plot goal shots (color by player)
    pitch.scatter(
        goal_shots['X'], 
        goal_shots['Y'], 
        s=goal_sizes,
        facecolors=goal_facecolor,
        edgecolors=goal_edgecolor,
        linewidth=2,
        ax=ax,
        zorder=3
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Goal',
               markerfacecolor='none', markeredgecolor=goal_edgecolor, markersize=10, linewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Shot (no goal)',
               markerfacecolor='none', markeredgecolor='black', markersize=10, linewidth=2)
    ]

    sample_xG = [0.05, 0.3, 0.7]
    sample_sizes = np.clip(np.array(sample_xG) * 1000, min_size, None)
    for xg_val, size in zip(sample_xG, sample_sizes):
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor='gray', markeredgecolor='gray',
                   label=f"xG = {xg_val}", markersize=np.sqrt(size)/2)
        )

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)

    # Player name label
    ax.text(40, 65, player_name, ha='center', va='center', fontsize=50, fontweight='bold', color='black', alpha=0.5)

    return fig


def get_two_players_radar_figure_matplotlib(df, attributes, player1, player2=None, pos_filter=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import percentileofscore

    # Filter DataFrame by position if given
    if pos_filter:
        df_filtered = df[df['pos'].str.contains(pos_filter, na=False, case=False)]
    else:
        df_filtered = df.copy()

    df_percentiles = df_filtered.copy()

    # Compute percentiles for each attribute
    for col in attributes:
        col_data = pd.to_numeric(df_filtered[col], errors='coerce').dropna().astype(float).values

        def get_percentile(val):
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                return 0
            if np.isnan(val_f):
                return 0
            return percentileofscore(col_data, val_f) / 100  # scale 0-1

        df_percentiles[col] = df[col].apply(get_percentile)

    def get_player_data(player_name):
        perc_row = df_percentiles[df_percentiles['player'] == player_name]
        raw_row = df[df['player'] == player_name]
        return perc_row, raw_row

    row1_percentile, row1_raw = get_player_data(player1)
    if row1_percentile.empty or row1_raw.empty:
        return None, f"Player **{player1}** does not play in the selected position or is not found."

    if player2:
        row2_percentile, row2_raw = get_player_data(player2)
        if row2_percentile.empty or row2_raw.empty:
            return None, f"Player **{player2}** does not play in the selected position or is not found."
    else:
        row2_percentile, row2_raw = None, None

    labels = attributes
    num_vars = len(labels)

    # Compute angle for each axis in the plot (in radians)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    # Get percentile values for players
    perc_values_1 = row1_percentile[attributes].values.flatten().tolist()
    perc_values_1 += perc_values_1[:1]

    if player2:
        perc_values_2 = row2_percentile[attributes].values.flatten().tolist()
        perc_values_2 += perc_values_2[:1]

    # Average percentiles across all players (for comparison)
    avg_percentiles = df_percentiles[attributes].mean().values.tolist()
    avg_percentiles += avg_percentiles[:1]

    # Create figure and polar subplot with gray background
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')

    # Draw labels on the axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', fontsize=13)

    # Remove y tick labels (no raw values)
    ax.set_yticks([])
    ax.set_ylim(0, 1)

    # Draw grid lines and customize colors
    ax.grid(color='lightgray', linewidth=1, linestyle='--')
    ax.spines['polar'].set_color('lightgray')

    # Plot player 1
    ax.plot(angles, perc_values_1, color='blue', linewidth=2, label=player1)
    ax.fill(angles, perc_values_1, color='blue', alpha=0.25)

    # Plot player 2 if present
    if player2:
        ax.plot(angles, perc_values_2, color='red', linewidth=2, label=player2)
        ax.fill(angles, perc_values_2, color='red', alpha=0.25)

    # Plot average with filled area (same alpha as players)
    ax.plot(angles, avg_percentiles, color='green', linewidth=2, linestyle='--', label='Average')
    ax.fill(angles, avg_percentiles, color='green', alpha=0.5)

    # Legend with black text
    # Legend with black text, placed at top right inside the plot
    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    for text in leg.get_texts():
        text.set_color('black')


    plt.tight_layout()
    return fig, None
