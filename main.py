!pip install catboost optuna xgboost lightgbm
import pandas as pd
import numpy as np
import networkx as nx
import optuna
import random
import warnings

# Модели для ансамбля
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')

print("=== ФАЗА 1: Ультра-быстрые данные и Новые Графовые Фичи ===")
heroes_df = pd.read_csv('/kaggle/input/data-fusion-2026-case-3/data_heroes.csv')
objects_df = pd.read_csv('/kaggle/input/data-fusion-2026-case-3/data_objects.csv')
dist_start_df = pd.read_csv('/kaggle/input/data-fusion-2026-case-3/dist_start.csv')
dist_matrix = pd.read_csv('/kaggle/input/data-fusion-2026-case-3/dist_objects.csv').values

num_objects = len(objects_df)
full_dist_matrix = np.zeros((num_objects + 1, num_objects + 1))
full_dist_matrix[1:, 1:] = dist_matrix
full_dist_matrix[0, 1:] = dist_start_df['dist_start'].values
full_dist_matrix[1:, 0] = dist_start_df['dist_start'].values

day_open_dict = dict(zip(objects_df['object_id'], objects_df['day_open']))
hero_list = list(zip(heroes_df['hero_id'], heroes_df['move_points']))
all_objects_set = set(range(1, num_objects + 1))

# Параметры
VISIT_COST = 100
MAX_DAYS = 7
REWARD = 500
HERO_COST = 2500

# Сложные графовые фичи
sigma = np.std(dist_matrix)
adj_matrix = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
np.fill_diagonal(adj_matrix, 0)
G = nx.from_numpy_array(adj_matrix)

pagerank_scores = nx.pagerank(G, alpha=0.85)
closeness_scores = nx.closeness_centrality(G, distance='weight') # Новая фича

max_pr = max(pagerank_scores.values())
max_cl = max(closeness_scores.values())

pr_features = {i+1: pagerank_scores[i] / max_pr for i in range(num_objects)}
cl_features = {i+1: closeness_scores[i] / max_cl for i in range(num_objects)}
pr_features[0] = 0.0
cl_features[0] = 0.0

dist_to_depot = full_dist_matrix[:, 0] # Новая фича

def extract_features_god_mode(current_node, target_node, current_day, remaining_points, hero_max_points, hero_id):
    """Извлечение расширенного набора признаков"""
    dist = full_dist_matrix[current_node, target_node]
    obj_day_open = day_open_dict[target_node]

    sim_day, sim_points, points_needed = current_day, remaining_points, dist
    while points_needed > sim_points:
        points_needed -= sim_points
        sim_day += 1
        sim_points = hero_max_points
    sim_points -= points_needed

    wait_days = 0
    if sim_day < obj_day_open:
        wait_days = obj_day_open - sim_day
        sim_day = obj_day_open
        sim_points = hero_max_points

    if sim_day > obj_day_open:
        return None, None

    is_last_move = 1 if 0 < sim_points <= VISIT_COST else 0
    actual_visit_cost = sim_points if is_last_move else VISIT_COST

    if sim_points < actual_visit_cost and not is_last_move:
        sim_day += 1
        sim_points = hero_max_points
        if sim_day > obj_day_open: return None, None
        actual_visit_cost = VISIT_COST

    sim_points -= actual_visit_cost
    if sim_points < 0: sim_points = 0

    total_spent = abs((current_day * hero_max_points - remaining_points) - (sim_day * hero_max_points - sim_points)) + 1

    # НОВЫЕ ФИЧИ В ВЕКТОРЕ
    features = [
        dist,                                  # 0
        wait_days,                             # 1
        is_last_move,                          # 2
        REWARD / total_spent,                  # 3 (ROI)
        pr_features[target_node],              # 4 (PageRank)
        cl_features[target_node],              # 5 (Closeness)
        dist_to_depot[target_node],            # 6 (Путь домой)
        1 if sim_day == obj_day_open else 0,   # 7 (Открыто сегодня)
        wait_days * hero_max_points,           # 8 (Потерянные очки на ожидание)
        MAX_DAYS - sim_day,                    # 9
        sim_points,                            # 10
        hero_max_points / 2000.0               # 11 (Сила героя)
    ]
    return features, (sim_day, sim_points)

print("=== ФАЗА 2: Сбор RL Опыта (Rollouts) ===")
training_data = []
for rollout in range(6): # 6 проходов для богатого датасета
    visited = set()
    noise_level = 0.15 * rollout
    for hero_id, max_p in hero_list:
        curr_day, rem_p, curr_node = 1, max_p, 0
        hero_trajectory = []
        while curr_day <= MAX_DAYS:
            avail = all_objects_set - visited
            if not avail: break
            candidates = []
            for obj in avail:
                feats, state = extract_features_god_mode(curr_node, obj, curr_day, rem_p, max_p, hero_id)
                if feats is not None:
                    # Стартовая эвристика для сбора данных
                    score = feats[3]*15 - feats[1]*3 + feats[2]*10 + feats[4]*5 - feats[0]*0.01
                    score += random.uniform(0, noise_level * abs(score))
                    candidates.append((score, obj, feats, state))
            if not candidates: break
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_obj, best_feats, best_state = candidates[0]
            hero_trajectory.append({'features': best_feats, 'reward': REWARD})
            visited.add(best_obj)
            curr_node, curr_day, rem_p = best_obj, best_state[0], best_state[1]

        cumulative_reward = 0
        for step in reversed(hero_trajectory):
            cumulative_reward += step['reward']
            training_data.append(step['features'] + [cumulative_reward])

df_train = pd.DataFrame(training_data)
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

print("=== ФАЗА 3: Обучение Ансамбля AI Моделей ===")
# Обучаем модели один раз, чтобы Optuna работала молниеносно
models = {
    'cat': CatBoostRegressor(iterations=200, depth=6, verbose=False),
    'xgb': xgb.XGBRegressor(n_estimators=150, max_depth=5, verbosity=0),
    'lgb': lgb.LGBMRegressor(n_estimators=150, num_leaves=31, verbose=-1),
    'ada': AdaBoostRegressor(n_estimators=50),
    'bayes': BayesianRidge() # Байесовская регрессия
}

for name, model in models.items():
    print(f"Обучение {name.upper()}...")
    model.fit(X_train, y_train)

def run_god_inference(weights, h_weights):
    """Мега-симулятор маршрутов с использованием ансамбля моделей и эвристик"""
    final_routes = []
    final_visited = set()

    for hero_id, max_p in hero_list:
        curr_day, rem_p, curr_node, hero_gold = 1, max_p, 0, 0
        hero_path = []

        while curr_day <= MAX_DAYS:
            avail = all_objects_set - final_visited
            if not avail: break

            X_cand, obj_list, state_list = [], [], []
            for obj in avail:
                feats, state = extract_features_god_mode(curr_node, obj, curr_day, rem_p, max_p, hero_id)
                if feats is not None:
                    X_cand.append(feats)
                    obj_list.append(obj)
                    state_list.append(state)

            if not X_cand: break
            X_cand_np = np.array(X_cand)

            # 1. Ансамблирование ML (Мягкое голосование регрессоров)
            blend_preds = np.zeros(len(X_cand))
            for m_name, model in models.items():
                blend_preds += weights[m_name] * model.predict(X_cand_np)

            # 2. Добавление умных Эвристик
            for idx, feats in enumerate(X_cand):
                is_last_move, roi, pagerank, dist_depot = feats[2], feats[3], feats[4], feats[6]
                blend_preds[idx] += (
                    (is_last_move * h_weights['last_move']) +
                    (roi * h_weights['roi']) +
                    (pagerank * h_weights['pr']) -
                    (dist_depot * h_weights['depot_penalty']) # штраф за уход далеко от базы
                )

            best_idx = np.argmax(blend_preds)
            best_obj = obj_list[best_idx]
            final_visited.add(best_obj)
            hero_path.append({'hero_id': hero_id, 'object_id': best_obj})
            hero_gold += REWARD
            curr_node, curr_day, rem_p = best_obj, state_list[best_idx][0], state_list[best_idx][1]

        if hero_path:
            for step in hero_path: step['hero_total_gold'] = hero_gold
            final_routes.extend(hero_path)

    # Economic Pruning (Увольняем убыточных героев в конце)
    df_res = pd.DataFrame(final_routes)
    if not df_res.empty:
        max_h = df_res['hero_id'].max()
        while max_h > 0:
            last_hero = df_res[df_res['hero_id'] == max_h]
            if last_hero.empty:
                max_h -= 1; continue
            if last_hero['hero_total_gold'].iloc[0] < HERO_COST:
                df_res = df_res[df_res['hero_id'] != max_h]
                max_h -= 1
            else: break

    total_gold = len(df_res) * REWARD if not df_res.empty else 0
    total_cost = df_res['hero_id'].max() * HERO_COST if not df_res.empty else 0
    return total_gold - total_cost, df_res

print("=== ФАЗА 4: Optuna Meta-Tuning (Поиск Идеального Ансамбля) ===")
def objective(trial):
    # Подбор весов для ML Моделей (Blender)
    ml_weights = {
        'cat': trial.suggest_float('w_cat', 0.0, 1.0),
        'xgb': trial.suggest_float('w_xgb', 0.0, 1.0),
        'lgb': trial.suggest_float('w_lgb', 0.0, 1.0),
        'ada': trial.suggest_float('w_ada', 0.0, 0.5), # Ада обычно слабее, вес меньше
        'bayes': trial.suggest_float('w_bayes', 0.0, 0.5)
    }
    # Нормализация весов ML
    total_ml = sum(ml_weights.values()) + 1e-9
    ml_weights = {k: v / total_ml for k, v in ml_weights.items()}

    # Подбор весов для Эвристик
    h_weights = {
        'last_move': trial.suggest_int('h_last_move', 0, 5000),
        'roi': trial.suggest_float('h_roi', 0.0, 1000.0),
        'pr': trial.suggest_float('h_pr', 0.0, 2000.0),
        'depot_penalty': trial.suggest_float('h_depot', 0.0, 5.0)
    }

    score, _ = run_god_inference(ml_weights, h_weights)
    return score

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
print("Optuna начала подбор... (Тестируем Ансамбль)")
study.optimize(objective, n_trials=150)

print(f"🏆 ЛУЧШИЙ VRPTW SCORE: {study.best_value}")
print("Идеальные параметры Блендинга:", study.best_params)

print("=== ФАЗА 5: Финальная Генерация Решения ===")
# Вытаскиваем лучшие веса
best = study.best_params
best_ml = {
    'cat': best['w_cat'], 'xgb': best['w_xgb'], 'lgb': best['w_lgb'],
    'ada': best['w_ada'], 'bayes': best['w_bayes']
}
total_ml = sum(best_ml.values()) + 1e-9
best_ml = {k: v / total_ml for k, v in best_ml.items()}

best_h = {
    'last_move': best['h_last_move'],
    'roi': best['h_roi'],
    'pr': best['h_pr'],
    'depot_penalty': best['h_depot']
}

final_score, df_final = run_god_inference(best_ml, best_h)

df_final[['hero_id', 'object_id']].to_csv('god_mode_submission.csv', index=False)
print("\n" + "🔥"*25)
print(f"ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
print(f"Собрано мельниц: {len(df_final)}")
print(f"ИТОГОВЫЙ СКОР: {final_score}")
print("Файл сохранен как 'god_mode_submission.csv'")
print("🔥"*25)
