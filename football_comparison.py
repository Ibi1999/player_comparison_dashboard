import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from function_utils import *  # Your utility functions
import io
from PIL import Image, ImageOps

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

comparison_type = "Players"
st.markdown("<small>Created by Ibrahim Oksuzoglu</small>", unsafe_allow_html=True)

if comparison_type == "Players":
    emoji_map = {"Players": "👥"}
    st.title(f"{emoji_map[comparison_type]} {comparison_type} Comparison Dashboard")

    # Load player names and positions dataframe early
    player_names_df = pd.read_pickle('player_names_pl.pkl')

    # Popular comparisons dictionary
    popular_comparisons = {
        "Custom": {
            "player1": "Bukayo Saka",
            "player2": "Marcus Rashford",
            "attribute": "Attacking",
            "position": "MF",
        },
        "Mohamed Salah vs Erling Haaland": {
            "player1": "Mohamed Salah",
            "player2": "Erling Haaland",
            "attribute": "Attacking",
            "position": "FW",
        },
        "Kevin De Bruyne vs Bruno Fernandes": {
            "player1": "Kevin De Bruyne",
            "player2": "Bruno Fernandes",
            "attribute": "Attacking",
            "position": "MF",
        },
        "Virgil van Dijk vs William Saliba": {
            "player1": "Virgil van Dijk",
            "player2": "William Saliba",
            "attribute": "Defensive",
            "position": "DF",
        },
        "Alisson Becker vs Ederson": {
            "player1": "Alisson Becker",
            "player2": "David Raya",
            "attribute": "Goalkeeping",
            "position": "GK",
        },
        "Trent Alexander-Arnold vs Andrew Robertson": {
            "player1": "Trent Alexander-Arnold",
            "player2": "Andrew Robertson",
            "attribute": "Defensive",
            "position": "DF",
        },
        "Alexander Isak vs Ollie Watkins": {
            "player1": "Alexander Isak",
            "player2": "Ollie Watkins",
            "attribute": "Attacking",
            "position": "MF",
        },
        "Moisés Caicedo vs Declan Rice": {
            "player1": "Moisés Caicedo",
            "player2": "Declan Rice",
            "attribute": "Defensive",
            "position": "MF",
        },
        "Bukayo Saka vs Marcus Rashford": {
            "player1": "Bukayo Saka",
            "player2": "Marcus Rashford",
            "attribute": "Attacking",
            "position": "MF",
        }
    }

    # Popular Comparisons dropdown (full width)
    selected_comparison = st.selectbox(
        "⭐ Popular player comparison presets",
        options=list(popular_comparisons.keys()),
        index=0
    )

    # Set defaults based on selection
    if selected_comparison != "Custom":
        preset = popular_comparisons[selected_comparison]
        default_player1 = preset["player1"]
        default_player2 = preset["player2"]
        default_attribute = preset["attribute"]
        default_position = preset["position"]
    else:
        default_player1 = "Bukayo Saka"
        default_player2 = "Marcus Rashford"
        default_attribute = "Attacking"
        default_position = "MF"

    attribute_options = ["Attacking", "Defensive", "Goalkeeping"]

    # 4 columns for attribute, player1, player2, position
    col_attr, col_p1, col_p2, col_pos = st.columns([1, 1.5, 1.5, 1.5])

    with col_attr:
        attribute_type = st.selectbox(
            "🔁 Select Attribute to Compare",
            options=attribute_options,
            index=attribute_options.index(default_attribute)
        )

    # Filter players based on attribute type
    if attribute_type == "Goalkeeping":
        valid_players_df = player_names_df[player_names_df['pos'].str.contains("GK", na=False)]
    else:
        valid_players_df = player_names_df[~player_names_df['pos'].str.contains("GK", na=False)]

    valid_players = valid_players_df['player'].dropna().unique().tolist()

    with col_p1:
        player1 = st.selectbox(
            "🔵 Select Player 1",
            options=valid_players,
            index=valid_players.index(default_player1) if default_player1 in valid_players else 0
        )

    # Get player1's positions (list)
    player1_pos_string = valid_players_df.loc[valid_players_df['player'] == player1, 'pos'].values[0]
    player1_positions = sorted(set(pos.strip() for pos in player1_pos_string.split(',')))

    with col_pos:
        position = st.selectbox(
            "🎯 Select Position",
            options=player1_positions,
            index=player1_positions.index(default_position) if default_position in player1_positions else 0
        )

    # Filter player 2 by position and exclude player 1
    player2_df = valid_players_df[
        (valid_players_df['player'] != player1) &
        (valid_players_df['pos'].str.contains(position, na=False))
    ]
    player2_options = ["None"] + sorted(player2_df['player'].unique().tolist())

    with col_p2:
        player2 = st.selectbox(
            "🔴 Select Player 2",
            options=player2_options,
            index=player2_options.index(default_player2) if default_player2 in player2_options else 0
        )
        player2 = None if player2 == "None" else player2

    # Load base dataframes
    all_shots_df = pd.read_pickle('all_shots_df.pkl')
    df_filename_map = {
        "Attacking": "attacking_attributes_df.pkl",
        "Defensive": "defensive_attributes_df.pkl",
        "Goalkeeping": "goalkeeping_attributes_df.pkl",
    }
    attr_filename_map = {
        "Attacking": "attacking_attributes.pkl",
        "Defensive": "defensive_attributes.pkl.pkl",
        "Goalkeeping": "goalkeeping_attributes.pkl",
    }

    df_standard = pd.read_pickle(df_filename_map[attribute_type])
    attributes_standard = pd.read_pickle(attr_filename_map[attribute_type])
    df_per90 = pd.read_pickle(df_filename_map[attribute_type].replace("_df.pkl", "90_df.pkl"))
    attributes_per90 = pd.read_pickle(attr_filename_map[attribute_type].replace(".pkl", "90.pkl"))

    st.markdown("---")

    if player1:
        col_shot, col_dist, col_radar = st.columns(3)

        with col_shot:
            st.subheader("🎯 Shot Maps")
            tab1, tab2 = st.tabs(["🔵 Player 1", "🔴 Player 2"])
            with tab1:
                fig_shot1 = plot_player_shots(player1, all_shots_df, player_number=1)
                if fig_shot1:
                    st.pyplot(fig_shot1)
                else:
                    st.info(f"No shots data available for {player1}.")

            with tab2:
                if player2 and player2 != player1:
                    fig_shot2 = plot_player_shots(player2, all_shots_df, player_number=2)
                    if fig_shot2:
                        st.pyplot(fig_shot2)
                    else:
                        st.info(f"No shots data available for {player2}.")
                else:
                    st.info("Select a second player to view their shot map.")

        with col_dist:
            st.subheader("📈 Distribution Plots")
            tab_density, tab_box, tab_hist = st.tabs(["🌊 Density", "📦 Box", "📊 Histogram"])

            def display_figure(fig):
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    buf.seek(0)
                    st.image(ImageOps.expand(Image.open(buf), border=30, fill='white'))

            with tab_box:
                fig_box = plot_two_players_boxplots_matplotlib(df_standard, attributes_standard, player1, player2, pos_filter=position)
                display_figure(fig_box)

            with tab_density:
                fig_density = plot_two_players_density_plots_matplotlib(df_standard, attributes_standard, player1, player2, pos_filter=position)
                display_figure(fig_density)

            with tab_hist:
                fig_hist = plot_two_players_histogram_plots_matplotlib(df_standard, attributes_standard, player1, player2, pos_filter=position)
                display_figure(fig_hist)

        with col_radar:
            st.subheader("📡 Radar Plot")
            radar_tab1, radar_tab2 = st.tabs(["⏱️ Per 90", "📊 Total"])

            with radar_tab2:
                fig_radar, error = get_two_players_radar_figure_matplotlib(
                    df_standard, attributes_standard, player1, player2, pos_filter=position
                )
                if error:
                    st.error(error)
                else:
                    display_figure(fig_radar)

            with radar_tab1:
                fig_radar_90, error_90 = get_two_players_radar_figure_matplotlib(
                    df_per90, attributes_per90, player1, player2, pos_filter=position
                )
                if error_90:
                    st.error(error_90)
                else:
                    display_figure(fig_radar_90)

        st.markdown("---")

        st.subheader("📋 Comparison Matrix")
        styled_matrix = get_two_players_comparison_matrix(df_standard, attributes_standard, player1, player2, pos_filter=position)
        if styled_matrix is not None:
            st.dataframe(styled_matrix, use_container_width=True)
