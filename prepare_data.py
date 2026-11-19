import pandas as pd

# 1. Raw dataset load karein
df = pd.read_csv("data/raw_jobs.csv")

# 2. Responsibilities aur Keywords ko mila kar 'job_description' banayein
df['job_description'] = (
    df['Title'].fillna('') + " (" + df['ExperienceLevel'].fillna('') + "): " +
    df['Responsibilities'].fillna('') + " " +
    df['Keywords'].fillna('')
)

# 3. Zaroori 4 columns select karein
clean_df = pd.DataFrame({
    'job_id': df['JobID'],
    'job_title': df['Title'],
    'job_description': df['job_description'],
    'required_skills': df['Skills']
})

# 4. Safai karein aur final file save karein
clean_df.dropna(inplace=True)
clean_df.drop_duplicates(subset=['job_id'], inplace=True)
clean_df.to_csv("data/jobs.csv", index=False)

print(f"Mubarak! data/jobs.csv ban gayi hai. Total Jobs: {len(clean_df)}")