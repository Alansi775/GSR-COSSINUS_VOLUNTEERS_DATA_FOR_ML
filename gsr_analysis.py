import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the correct path to your CSV files
csv_dir = os.path.expanduser("~/Downloads/gsr")  # Expands to your full Mac path

# Get the list of CSV files in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Load and normalize data for each volunteer
all_volunteer_data = []

for i, file in enumerate(csv_files):
    file_path = os.path.join(csv_dir, file)  # Full path to CSV file
    df = pd.read_csv(file_path)
    volunteer_name = f'Volunteer {i+1}'
    df['Volunteer'] = volunteer_name

    # Normalize Resistance independently
    resistance_range = df['Resistance'].max() - df['Resistance'].min()
    if resistance_range == 0:
        df['Normalized Resistance'] = 0  # Handle edge case where all values are identical
    else:
        df['Normalized Resistance'] = (df['Resistance'] - df['Resistance'].min()) / (resistance_range + 1e-6)
        df['Normalized Resistance'] = 0.1 + df['Normalized Resistance'] * 0.9  # Rescale to 0.1–1 range

    all_volunteer_data.append(df)

# Combine all volunteers' data into a single DataFrame
data = pd.concat(all_volunteer_data, ignore_index=True)

# Define volunteers for different stages and female volunteers
cold_pressure_volunteers = [f'Volunteer {i}' for i in range(1, 17)]
doctor_game_volunteers = [f'Volunteer {i}' for i in range(17, 35)]
female_volunteers = {'Volunteer 10', 'Volunteer 12', 'Volunteer 13', 'Volunteer 14', 'Volunteer 15', 
                     'Volunteer 16', 'Volunteer 17', 'Volunteer 18', 'Volunteer 19', 'Volunteer 25', 'Volunteer 26'}

# Standardize bar width
total_volunteers = len(cold_pressure_volunteers) + len(doctor_game_volunteers)
bar_width = 4 / total_volunteers

# Function to plot graphs
def plot_graph(volunteers, title, description, color):
    fig, ax = plt.subplots(figsize=(11, 7))

    for i, volunteer in enumerate(volunteers):
        volunteer_data = data[data['Volunteer'] == volunteer]

        # Compute means for each stage
        relaxation_mean = volunteer_data[volunteer_data['Stage'] == 'Relaxation']['Normalized Resistance'].mean()
        stress_mean = volunteer_data[volunteer_data['Stage'] == 'Under Stress']['Normalized Resistance'].mean()

        # Plot bars
        ax.bar(i - bar_width / 2, relaxation_mean, width=bar_width, color='lightgray', edgecolor='black', label='Relaxation' if i == 0 else '')
        ax.bar(i + bar_width / 2, stress_mean, width=bar_width, color=color, edgecolor='black', label='Under Stress' if i == 0 else '')

        # Annotate female volunteers
        if volunteer in female_volunteers:
            ax.text(i, max(relaxation_mean, stress_mean) + 0.03, 'Female', ha='center', color='black', fontsize=10)

    ax.set_xticks(range(len(volunteers)))
    ax.set_xticklabels(volunteers, fontsize=8, rotation=45, ha='right')
    ax.set_xlabel('Volunteer')
    ax.set_ylabel('Normalized Resistance')
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True)

    plt.figtext(0.5, -0.05, description, wrap=True, horizontalalignment='center', fontsize=10)
    plt.tight_layout()
    plt.show()

# Plot graphs
plot_graph(
    cold_pressure_volunteers, 
    "Stress Levels (GSR) for Cold Pressure Volunteers", 
    "Volunteers 1-16 worked with Cold Pressure in the Under Stress stage.", 
    'gray'
)

plot_graph(
    doctor_game_volunteers, 
    "Stress Levels (GSR) for Doctor Game Volunteers", 
    "Volunteers 17-34 worked with the Doctor Game in the Under Stress stage.", 
    'darkgray'
)