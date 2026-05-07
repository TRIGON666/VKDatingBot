#!/usr/bin/env python3
"""
Интеграционный тест для проверки компонентов NLP.

Запуск: python test_nlp_integration.py
"""

import os
import sys
import traceback

from src.config import NLP_DATA_PATH, NLP_MODEL_PATH

# Проверяет доступность NLP-модулей и символов.
def test_imports():
    """Проверить что все модули импортируются"""
    print("\n" + "=" * 60)
    print("[ТЕСТ 1] Импорт модулей")
    print("=" * 60)
    
    modules = [
        ("src.nlp_preprocessing", ["preprocess_profile", "extract_keywords"]),
        ("src.nlp_data_collector", ["log_interaction_for_nlp", "get_nlp_stats"]),
        ("src.nlp_metrics", ["NLPMetricsTracker"]),
        ("src.nlp_compatibility", ["predict_text_compatibility", "train_text_compatibility_model"]),
    ]
    
    all_ok = True
    for module_name, expected_symbols in modules:
        try:
            module = __import__(module_name, fromlist=expected_symbols)
            for symbol in expected_symbols:
                if not hasattr(module, symbol):
                    print("ОШИБКА: {}.{} - НЕ НАЙДЕН".format(module_name, symbol))
                    all_ok = False
                else:
                    print("ОК: {}.{}".format(module_name, symbol))
        except Exception as e:
            print("ОШИБКА: {} - {}".format(module_name, str(e)[:50]))
            all_ok = False
    
    assert all_ok

# Проверяет обработку текста и выделение ключевых слов.
def test_preprocessing():
    """Тест предварительной обработки текста"""
    print("\n" + "=" * 60)
    print("[ТЕСТ 2] Предварительная обработка")
    print("=" * 60)
    
    try:
        from src.nlp_preprocessing import preprocess_profile, extract_keywords
        
        test_text = "Люблю путешествовать и походы"
        processed = preprocess_profile(test_text)
        
        print("Входной текст: {}".format(test_text))
        print("Обработанный: {}".format(processed))
        print("ОК: Обработка работает")
        
        keywords = extract_keywords("Люблю путешествовать", top_n=3)
        print("Ключевые слова: {}".format(keywords))
        print("ОК: Извлечение ключевых слов работает")
        
        assert processed
        assert isinstance(keywords, list)
    except Exception as e:
        print("ОШИБКА: Ошибка обработки: {}".format(str(e)[:50]))
        traceback.print_exc()
        raise AssertionError("Preprocessing test failed") from e

# Проверяет расчет NLP-метрик.
def test_metrics():
    """Тест отслеживания метрик"""
    print("\n" + "=" * 60)
    print("[ТЕСТ 3] Отслеживание метрик")
    print("=" * 60)
    
    try:
        from src.nlp_metrics import NLPMetricsTracker
        
        tracker = NLPMetricsTracker()
        print("ОК: NLPMetricsTracker инициализирован")
        
        # Try to get metrics (may be empty)
        metrics = tracker.calculate_metrics(hours=None)
        print("ОК: Расчет метрик работает")
        print("   Предсказаний записано: {}".format(metrics.get('predictions_count', 0)))
        
        assert isinstance(metrics, dict)
    except Exception as e:
        print("ОШИБКА: Ошибка метрик: {}".format(str(e)[:50]))
        traceback.print_exc()
        raise AssertionError("Metrics test failed") from e

# Проверяет чтение статистики собранных NLP-данных.
def test_data_collector():
    """Тест сбора данных"""
    print("\n" + "=" * 60)
    print("[ТЕСТ 4] Сбор данных")
    print("=" * 60)
    
    try:
        from src.nlp_data_collector import get_nlp_stats
        
        if os.path.exists(NLP_DATA_PATH):
            stats = get_nlp_stats(NLP_DATA_PATH)
            print("ОК: Сборщик данных работает")
            print("   Всего примеров: {}".format(stats['total']))
            print("   Положительных: {}".format(stats['positive']))
            print("   Нейтральных: {}".format(stats['neutral']))  
            print("   Отрицательных: {}".format(stats['negative']))
        else:
            print("ПРОПУСК: Файл данных не создан (нормально - бот еще не работал)")
        
        # Если исключений нет, проверка считается успешной.
    except Exception as e:
        print("ОШИБКА: Ошибка сборщика: {}".format(str(e)[:50]))
        traceback.print_exc()
        raise AssertionError("Data collector test failed") from e

# Проверяет загрузку модели и базовый прогноз.
def test_model_loading():
    """Тест загрузки модели"""
    print("\n" + "=" * 60)
    print("[ТЕСТ 5] Загрузка модели")
    print("=" * 60)
    
    try:
        from src.nlp_compatibility import predict_text_compatibility
        
        if not os.path.exists(NLP_MODEL_PATH):
            print("ПРОПУСК: Модель не обучена (нормально - бот еще не работал)")
            return
        
        result = predict_text_compatibility(
            "Люблю путешествовать",
            "Путешествия это жизнь",
            model_path=NLP_MODEL_PATH,
        )
        
        print("ОК: Модель загружена и работает")
        print("   Предсказание: {}".format(result))
        
        assert "label" in result
    except Exception as e:
        print("ОШИБКА: Ошибка загрузки модели: {}".format(str(e)[:50]))
        traceback.print_exc()
        raise AssertionError("Model loading test failed") from e

# Проверяет наличие CLI-скриптов NLP.
def test_script_availability():
    """Проверить что все скрипты управления есть"""
    print("\n" + "=" * 60)
    print("[ТЕСТ 6] Доступность скриптов")
    print("=" * 60)
    
    scripts = [
        "train_nlp_model.py",
        "monitor_nlp.py",
        "auto_train_nlp.py",
        "analyze_nlp.py",
        "nlp_utils.py",
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print("ОК: {}".format(script))
        else:
            print("ОШИБКА: {} - НЕ НАЙДЕН".format(script))
            all_exist = False
    
    assert all_exist

# Точка входа: разбирает аргументы и запускает нужный сценарий.
def main():
    print("\n" + "=" * 60)
    print("[НАБОР ТЕСТОВ] Интеграционные тесты NLP")
    print("=" * 60)
    
    tests = [
        ("Импорт модулей", test_imports),
        ("Предварительная обработка", test_preprocessing),
        ("Метрики", test_metrics),
        ("Сбор данных", test_data_collector),
        ("Загрузка модели", test_model_loading),
        ("Скрипты", test_script_availability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True if result is None else bool(result)))
        except Exception as e:
            print("\nОШИБКА: Тест '{}' завалился: {}".format(test_name, str(e)[:50]))
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("[ИТОГИ] Результаты тестирования")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[ОК]".ljust(8) if result else "[ОШИБКА]".ljust(8)
        print("{} {}".format(status, test_name))
    
    print("\nВсего: {}/{} тестов пройдено".format(passed, total))
    print("=" * 60)
    
    if passed == total:
        print("\nУСПЕХ: ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к использованию.")
        return 0
    else:
        print("\nПРЕДУПРЕЖДЕНИЕ: {} тест(ов) не пройдено. Проверьте ошибки выше.".format(total - passed))
        return 1

if __name__ == "__main__":
    sys.exit(main())
