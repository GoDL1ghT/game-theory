import numpy as np
import pandas as pd
import graphviz
from IPython.display import display

# 1. Настройка данных и параметров
np.random.seed(42)

# Структура дерева решений (3 уровня)
data = {
    'Быстрый релиз': {
        'Стабильная нагрузка': {'prob': 0.6, 'Успех': (0.8, 1000), 'Частичный успех': (0.2, 500), 'Провал': (0.0, 0)},
        'Пиковая нагрузка': {'prob': 0.3, 'Успех': (0.3, 800), 'Частичный успех': (0.4, 300), 'Провал': (0.3, -200)},
        'Сбой зависимости': {'prob': 0.1, 'Успех': (0.1, 200), 'Частичный успех': (0.2, -100), 'Провал': (0.7, -1000)}
    },
    'Тестирование': {
        'Стабильная нагрузка': {'prob': 0.7, 'Успех': (0.9, 900), 'Частичный успех': (0.1, 600), 'Провал': (0.0, 0)},
        'Пиковая нагрузка': {'prob': 0.25, 'Успех': (0.7, 700), 'Частичный успех': (0.2, 400), 'Провал': (0.1, -100)},
        'Сбой зависимости': {'prob': 0.05, 'Успех': (0.5, 400), 'Частичный успех': (0.3, 100), 'Провал': (0.2, -300)}
    },
    'Отмена': {
        'Стабильная нагрузка': {'prob': 1.0, 'Провал': (1.0, -50)}
    }
}

# 2. Функции для расчета и детализации
def get_strategy_details(strategy_name, strategy_data):
    all_outcomes = []
    all_probs = []
    
    print(f"\n--- Детализация стратегии: {strategy_name} ---")
    for env_state, env_val in strategy_data.items():
        p_env = env_val['prob']
        for res_key, res_tuple in env_val.items():
            if res_key == 'prob': continue
            p_res, value = res_tuple
            if p_res > 0:
                combined_p = p_env * p_res
                print(f"  • {env_state} -> {res_key}: Вероятность {combined_p:.2f}, Выигрыш {value}$")
                all_outcomes.append(value)
                all_probs.append(combined_p)
    
    outcomes = np.array(all_outcomes)
    probs = np.array(all_probs)
    emv = np.sum(outcomes * probs)
    std_dev = np.sqrt(np.sum(probs * (outcomes - emv)**2))
    return emv, std_dev

# 3. Расчет и вывод результатов
results = []
for strategy in data:
    emv, std = get_strategy_details(strategy, data[strategy])
    results.append({'Стратегия': strategy, 'EMV': emv, 'Sigma (Risk)': std})

print("\nСВОДНАЯ ТАБЛИЦА:")
display(pd.DataFrame(results))

# 4. Визуализация дерева
dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir':'LR', 'size':'10,10'})
dot.node('root', 'Выбор стратегии', shape='box', style='filled', fillcolor='#DCDCDC')
colors = {'Быстрый релиз': '#FFCCCC', 'Тестирование': '#CCFFCC', 'Отмена': '#CCCCFF'}

for strategy, env_states in data.items():
    dot.node(strategy, strategy, shape='box', style='filled', fillcolor=colors.get(strategy, 'white'))
    dot.edge('root', strategy)
    for env_state, env_val in env_states.items():
        env_id = f"{strategy}_{env_state}"
        dot.node(env_id, f"{env_state}\n(p={env_val['prob']})", shape='ellipse')
        dot.edge(strategy, env_id)
        for res_key, res_data in env_val.items():
            if res_key != 'prob' and res_data[0] > 0:
                p_res, val = res_data
                res_id = f"{env_id}_{res_key}"
                dot.node(res_id, f"{res_key}\n{val}$ (p={p_res})", shape='plaintext')
                dot.edge(env_id, res_id)

display(dot)

# ВЫВОДЫ:
# 1. Проведен детальный расчет каждого из возможных сценариев, включая комбинацию состояний среды и конечных результатов.
# 2. Выигрыши в сценарии 'Быстрый релиз' варьируются от -1000$ до 1000$, что объясняет высокую волатильность и риск.
# 3. Сценарий 'Тестирование' сужает диапазон возможных исходов (-300$ до 900$), обеспечивая более предсказуемый финансовый результат.
# 4. Несмотря на наличие отрицательных исходов, EMV стратегии 'Тестирование' остается максимальным.
