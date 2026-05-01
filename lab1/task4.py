import numpy as np
from fractions import Fraction

# Константы для проверки согласованности (шкала Саати для n от 1 до 10)
RI_DICT = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}


def get_priority_vector(matrix):
    """Расчет собственного вектора (вектора приоритетов) методом нормализации."""
    column_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / column_sums
    priority_vector = np.mean(normalized_matrix, axis=1)
    return priority_vector


def check_consistency(matrix, n):
    """Проверка индекса (CI) и отношения согласованности (CR)."""
    if n <= 2:
        return 0.0, 0.0
    
    priority_vector = get_priority_vector(matrix)
    weighted_sum_vector = np.dot(matrix, priority_vector)
    lambda_max = np.mean(weighted_sum_vector / priority_vector)
    
    ci = (lambda_max - n) / (n - 1)
    cr = ci / RI_DICT[n]
    return ci, cr


def input_comparison_matrix(labels, title):
    """Консольный ввод матрицы парных сравнений с поддержкой дробей."""
    n = len(labels)
    while True:
        print(f"\n--- Заполнение матрицы: {title} ---")
        print("Используйте шкалу Саати (1-9). Например: 5, 0.2 или 1/3.")
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                while True:
                    try:
                        raw_val = input(f"Сравните '{labels[i]}' с '{labels[j]}': ").strip()
                        val = float(Fraction(raw_val))
                        if val <= 0:
                            raise ValueError
                        matrix[i, j] = val
                        matrix[j, i] = 1 / val
                        break
                    except (ValueError, ZeroDivisionError):
                        print("Ошибка: введите положительное число или дробь (например, 1/3).")
        
        ci, cr = check_consistency(matrix, n)
        print(f"Результаты проверки: CI = {ci:.4f}, CR = {cr:.4f}")
        
        if cr <= 0.1:
            print("Статус: Матрица согласована.")
            return matrix, get_priority_vector(matrix), ci, cr
        else:
            print("Статус: Матрица НЕ согласована (CR > 0.1). Пожалуйста, повторите ввод.")


def main():
    print("Цель: Выбор оптимальной СУБД")
    
    # 1. Определение альтернатив и критериев
    alternatives = [
        "PostgreSQL", "MongoDB", "Redis", 
        "Cassandra", "MySQL", "DynamoDB"
    ]
    criteria = [
        "Производительность", "Стоимость", 
        "Масштабируемость", "Простота поддержки"
    ]
    
    # 2. Сравнение критериев относительно цели
    criteria_matrix, criteria_weights, c_ci, c_cr = input_comparison_matrix(
        criteria, "Критерии относительно Цели"
    )
    
    # 3. Сравнение альтернатив по каждому критерию
    alt_weights_per_criterion = []
    all_stats = []
    
    for criterion in criteria:
        matrix, weights, ci, cr = input_comparison_matrix(
            alternatives, f"Альтернативы по критерию '{criterion}'"
        )
        alt_weights_per_criterion.append(weights)
        all_stats.append((criterion, ci, cr, weights))
    
    # 4. Расчет глобальных приоритетов
    alt_weights_matrix = np.column_stack(alt_weights_per_criterion)
    global_priorities = np.dot(alt_weights_matrix, criteria_weights)
    
    # 5. Вывод результатов
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ РАСЧЕТА")
    print("="*50)
    
    print(f"\nВеса критериев: {dict(zip(criteria, np.round(criteria_weights, 4)))}")
    print(f"Согласованность критериев: CI={c_ci:.4f}, CR={c_cr:.4f}")
    
    print("\nЛокальные веса альтернатив по критериям:")
    for crit, ci, cr, w in all_stats:
        print(f"- {crit}: CI={ci:.4f}, CR={cr:.4f}")
        for idx, alt in enumerate(alternatives):
            print(f"  {alt}: {w[idx]:.4f}")
            
    print("\nГлобальные приоритеты СУБД:")
    results = sorted(zip(alternatives, global_priorities), key=lambda x: x[1], reverse=True)
    for name, score in results:
        print(f"{name}: {score:.4f}")
    
    print(f"\nЛучший выбор: {results[0][0]}")

if __name__ == "__main__":
    np.random.seed(42)
    main()

# ВЫВОДЫ:
#    1. Применен метод анализа иерархий для выбора СУБД из 6 альтернатив по 4 критериям.
#    2. Обеспечена согласованность всех матриц суждений (CR <= 0.1), что гарантирует логичность выбора.
#    3. Приоритетным критерием в данном расчете является '{criteria[np.argmax(criteria_weights)]}'.
#    4. На основе математической модели лучшей СУБД признана {results[0][0]}.
#    5. Код поддерживает ввод в дробном формате, что упрощает заполнение обратно-симметричных матриц.
