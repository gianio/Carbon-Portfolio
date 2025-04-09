import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns  # For potentially more appealing styles

# Set Streamlit page config with light green background via HTML/CSS
st.markdown("""
    <style>
    body {
        background-color: #e6f2e6;
    }
    .stApp {
        background-color: #e6f2e6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Multi-Year Carbon Portfolio Builder")

# Upload CSV
df = st.file_uploader("Upload project CSV", type="csv")

if df:
    data = pd.read_csv(df)

    st.subheader("Project Overview")
    years = st.slider("How many years should the portfolio span?", 1, 6, 6)
    end_year = 2025 + years - 1
    selected_years = list(range(2025, end_year + 1))

    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in data.columns]
    overview['avg_price'] = overview[price_cols].mean(axis=1)
    if 'description' in overview.columns:
        st.dataframe(overview[['project name', 'project type', 'description', 'avg_price']].drop_duplicates())
    else:
        st.dataframe(overview[['project name', 'project type', 'avg_price']].drop_duplicates())

    st.subheader("Step 1: Define Portfolio Settings")
    annual_volumes = {}
    volume_data = []
    for year in selected_years:
        volume_data.append({"year": year, "volume": 1000})  # Default volume
    volumes_df = pd.DataFrame(volume_data)
    edited_volumes_df = st.data_editor(volumes_df, num_rows="dynamic", column_config={"year": st.column_config.Column(disabled=True)})
    annual_volumes = edited_volumes_df.set_index('year')['volume'].to_dict()

    removal_target = st.slider(f"Target total removal % for year {end_year}", 0, 100, 80) / 100
    transition_speed = st.slider("Transition Speed (1 = Slow, 10 = Fast)", 1, 10, 5)
    removal_preference = st.slider("Removal Preference (1 = Natural, 10 = Technical)", 1, 10, 5)

    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = data['project name'].unique().tolist()
    selected_projects = st.multiselect("Select projects to include in the portfolio:", project_names)
    favorite_projects = st.multiselect("Select your favorite projects (will get a +10% priority boost):", selected_projects)

    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()

        # Apply favorite project boost
        if favorite_projects:
            selected_df['priority'] = selected_df.apply(
                lambda row: row['priority'] + 10 if row['project name'] in favorite_projects and pd.notna(row['priority']) and row['priority'] < 100 else row['priority'],
                axis=1
            )
            # Ensure priority doesn't exceed 100%
            selected_df['priority'] = selected_df['priority'].clip(upper=100)

        types = ['technical removal', 'natural removal', 'reduction']
        project_types = {t: selected_df[selected_df['project type'] == t] for t in types}

        removal_percentages = []
        for i, year in enumerate(selected_years):
            progress = (i + 1) / years
            factor = 0.5 + 0.1 * transition_speed
            removal_pct = removal_target * (progress ** factor)
            removal_percentages.append(min(removal_pct, removal_target))

        category_split = {
            'technical removal': removal_preference / 10,
            'natural removal': 1 - (removal_preference / 10)
        }

        portfolio = {year: {} for year in selected_years}
        broken_rules = []

        for year_idx, year in enumerate(selected_years):
            year_str = f"{year}"
            removal_share = removal_percentages[year_idx]
            reduction_share = 1 - removal_share
            annual_volume = annual_volumes.get(year, 0)

            volumes = {}
            total_allocated = 0

            for category in types:
                if category == 'reduction':
                    category_share = reduction_share
                else:
                    category_share = removal_share * category_split[category]

                category_projects = project_types[category].copy()
                category_projects = category_projects[category_projects.get(f"available volume {year_str}", 0) > 0]

                if category_projects.empty:
                    broken_rules.append(f"No available volume for {category} in {year}.")
                    continue

                allocated_category_volume = 0

                if 'priority' in category_projects.columns:
                    priority_projects = category_projects[category_projects['priority'].notna()].copy()
                    remaining_projects = category_projects[category_projects['priority'].isna()].copy()

                    # Allocate based on priority
                    for _, row in priority_projects.iterrows():
                        priority = row['priority'] / 100.0  # Convert percentage to fraction
                        target_volume = annual_volume * category_share * priority
                        max_available = row.get(f"available volume {year_str}", 0)
                        vol = min(target_volume, max_available)
                        volumes[row['project name']] = {
                            'volume': int(vol),
                            'price': row.get(f'price {year_str}', 0),
                            'type': category
                        }
                        allocated_category_volume += vol

                    # Allocate remaining volume equally among non-priority projects
                    num_remaining = len(remaining_projects)
                    if num_remaining > 0:
                        remaining_category_volume = annual_volume * category_share - allocated_category_volume
                        if remaining_category_volume > 0:
                            share_per_remaining = remaining_category_volume / num_remaining
                            for _, row in remaining_projects.iterrows():
                                max_available = row.get(f"available volume {year_str}", 0)
                                vol = min(share_per_remaining, max_available)
                                volumes[row['project name']] = {
                                    'volume': int(vol),
                                    'price': row.get(f'price {year_str}', 0),
                                    'type': category
                                }
                                allocated_category_volume += vol
                else:
                    # If 'priority' column doesn't exist for this category, allocate equally
                    num_projects = len(category_projects)
                    if num_projects > 0:
                        share_per_project = annual_volume * category_share / num_projects
                        for _, row in category_projects.iterrows():
                            max_available = row.get(f"available volume {year_str}", 0)
                            vol = min(share_per_project, max_available)
                            volumes[row['project name']] = {
                                'volume': int(vol),
                                'price': row.get(f'price {year_str}', 0),
                                'type': category
                            }
                            allocated_category_volume += vol

                total_allocated += allocated_category_volume

            if total_allocated > 0:
                scale_factor = annual_volume / total_allocated
                for v in volumes.values():
                    v['volume'] = int(v['volume'] * scale_factor)
            else:
                broken_rules.append(f"No available projects in {year}, cannot allocate volume.")

            portfolio[year] = volumes

        composition_by_type = {t: [] for t in types}
        avg_prices = []

        for year in selected_years:
            annual_volume = annual_volumes[year]
            totals = {t: 0 for t in types}
            total_cost = 0
            for data in portfolio[year].values():
                totals[data['type']] += data['volume']
                total_cost += data['volume'] * data['price']

            for t in types:
                composition_by_type[t].append(totals[t])

            avg_prices.append(total_cost / annual_volume if annual_volume > 0 else 0)

        st.subheader("Portfolio Composition & Price Over Time")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for composition
        fig.add_trace(go.Bar(x=selected_years, y=composition_by_type['technical removal'], name='Technical Removal', marker_color='#8BC34A'), secondary_y=False)
        fig.add_trace(go.Bar(x=selected_years, y=composition_by_type['natural removal'], name='Natural Removal', marker_color='#AED581'), secondary_y=False)
        fig.add_trace(go.Bar(x=selected_years, y=composition_by_type['reduction'], name='Reduction', marker_color='#C5E1A5'), secondary_y=False)
        
        # Line plot for average price
        fig.add_trace(go.Scatter(x=selected_years, y=avg_prices, name='Average Price', marker=dict(symbol='circle'), line=dict(color='#558B2F')), secondary_y=True)
        
        # Update layout for better aesthetics
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Volume',
            yaxis2_title='Average Price (€)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='stack',
            template="plotly_white", # Use a clean template
            plot_bgcolor='#9ec143',  # Set the plot background color
            paper_bgcolor='#9ec143'     # Set the paper (surrounding) background color
        )
        
        st.plotly_chart(fig)

        final_tech = composition_by_type['technical removal'][-1]
        final_nat = composition_by_type['natural removal'][-1]
        final_total = final_tech + final_nat + composition_by_type['reduction'][-1]
        achieved_removal = (final_tech + final_nat) / final_total if final_total > 0 else 0

        st.markdown(f"**Achieved Removal % in {end_year}: {achieved_removal * 100:.2f}%**")

        if broken_rules:
            st.warning("One or more constraints could not be fully satisfied:")
            for msg in broken_rules:
                st.text(f"- {msg}")

        if st.checkbox("Show raw project allocations"):
            full_table = []
            for year, projects in portfolio.items():
                for name, info in projects.items():
                    full_table.append({
                        'year': year,
                        'project name': name,
                        'type': info['type'],
                        'volume': info['volume'],
                        'price': info['price'],
                        'cost': info['volume'] * info['price']
                    })
            st.dataframe(pd.DataFrame(full_table))
