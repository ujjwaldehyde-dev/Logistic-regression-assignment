"""
Generates a synthetic Titanic passenger dataset that mirrors the well-known
statistical structure of the real Kaggle Titanic dataset (891 passengers,
same columns, same broad survival patterns by class/sex/age). This project
runs in an offline sandbox with no internet access, so the real Kaggle CSV
could not be downloaded — this script builds a statistically realistic stand-in
so the full ML pipeline below is genuine, reproducible, end-to-end work.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 891

# Pclass: 3rd class most common historically (~55%), then 1st (~24%), 2nd (~21%)
pclass = rng.choice([1, 2, 3], size=N, p=[0.242, 0.207, 0.551])

# Sex: ~65% male, 35% female (matches historical Titanic manifest roughly)
sex = rng.choice(['male', 'female'], size=N, p=[0.647, 0.353])

# Age: depends loosely on class (1st class skews older), with missing values (~20%)
age_base = np.where(pclass == 1, 38, np.where(pclass == 2, 30, 25)).astype(float)
age = rng.normal(age_base, 13).clip(0.42, 80)
age_missing_mask = rng.random(N) < 0.198
age_full = age.copy()
age[age_missing_mask] = np.nan

# SibSp / Parch: mostly 0, occasionally 1-4
sibsp = rng.choice([0, 1, 2, 3, 4, 5, 8], size=N, p=[0.679, 0.235, 0.032, 0.018, 0.008, 0.005, 0.023])
parch = rng.choice([0, 1, 2, 3, 4, 5, 6], size=N, p=[0.760, 0.132, 0.08, 0.006, 0.006, 0.005, 0.011])

family_size = sibsp + parch + 1

# Fare: correlated with class, with some noise, right-skewed
fare_base = np.where(pclass == 1, 84, np.where(pclass == 2, 20.5, 13.7))
fare = rng.gamma(shape=2.0, scale=fare_base / 2.0)
fare = np.round(fare, 4)

# Embarked: S most common, then C, then Q; couple missing
embarked = rng.choice(['S', 'C', 'Q'], size=N, p=[0.724, 0.189, 0.087])
embarked = embarked.astype(object)
emb_missing_idx = rng.choice(N, size=2, replace=False)
for i in emb_missing_idx:
    embarked[i] = np.nan

# Cabin: mostly missing (matches real dataset's ~77% missing), else a deck letter+number
cabin = np.array([np.nan] * N, dtype=object)
has_cabin = rng.random(N) < 0.23
decks = np.array(list('ABCDEFG'))
for i in np.where(has_cabin)[0]:
    deck = rng.choice(decks, p=[0.06, 0.16, 0.22, 0.20, 0.16, 0.12, 0.08])
    cabin[i] = f"{deck}{rng.integers(1, 148)}"

# Titles (feature engineering source) based on sex/age for realistic name-derived titles
def make_title(s, a):
    if s == 'male':
        return 'Master' if (not np.isnan(a) and a < 14) else rng.choice(['Mr', 'Dr', 'Rev', 'Major'], p=[0.93, 0.03, 0.02, 0.02])
    else:
        return rng.choice(['Mrs', 'Miss'], p=[0.42, 0.58])

titles = [make_title(sex[i], age_full[i]) for i in range(N)]

first_names_m = ['James','John','Robert','William','Charles','George','Frank','Edward','Henry','Thomas']
first_names_f = ['Mary','Anna','Elizabeth','Margaret','Alice','Florence','Emily','Helen','Ida','Nora']
last_names = ['Smith','Johnson','Williams','Brown','Jones','Miller','Davis','Wilson','Moore','Taylor',
              'Anderson','Thomas','Jackson','White','Harris','Martin','Thompson','Garcia','Martinez','Robinson']

names = []
for i in range(N):
    ln = rng.choice(last_names)
    fn = rng.choice(first_names_m) if sex[i] == 'male' else rng.choice(first_names_f)
    names.append(f"{ln}, {titles[i]}. {fn}")

ticket = [f"{rng.integers(10000,999999)}" for _ in range(N)]

# ---- Survival probability model (this encodes the well-documented "women and
# children first" + class-privilege pattern, plus logistic noise) ----
logit = (
    -1.75
    + 2.6 * (sex == 'female')
    + 1.15 * (pclass == 1)
    + 0.35 * (pclass == 2)
    - 0.9 * (pclass == 3)
    + 0.03 * (fare - fare.mean()) / fare.std()
    - 0.55 * ((age_full < 14) == False) * 0  # placeholder, replaced below
)
child_bonus = np.where(age_full < 14, 1.1, 0.0)
family_penalty = np.where(family_size > 4, -0.6, np.where(family_size == 1, -0.15, 0.15))
logit = logit + child_bonus + family_penalty
prob = 1 / (1 + np.exp(-logit))
survived = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    'PassengerId': np.arange(1, N + 1),
    'Survived': survived,
    'Pclass': pclass,
    'Name': names,
    'Sex': sex,
    'Age': np.round(age, 2),
    'SibSp': sibsp,
    'Parch': parch,
    'Ticket': ticket,
    'Fare': fare,
    'Cabin': cabin,
    'Embarked': embarked,
})

df.to_csv('/home/claude/titanic_project/data/titanic.csv', index=False)
print(df.shape)
print(df['Survived'].mean())
print(df.isna().sum())
