from urbanomy.methods.agent_interface.internal.district_optimization.intents import (
    looks_like_district_optimization_request,
    parse_district_optimization_intent,
)


def test_pop_size_symbolic_name_question():
    intent = parse_district_optimization_intent("Какое значение pop_size в оптимизаторе?")

    assert intent.kind == "algorithm_parameter"
    assert intent.parameter_name == "pop_size"


def test_pop_size_russian_alias_question():
    intent = parse_district_optimization_intent("Какой размер популяции?")

    assert intent.kind == "algorithm_parameter"
    assert intent.parameter_name == "pop_size"


def test_n_gen_russian_alias_question():
    intent = parse_district_optimization_intent("Сколько поколений у оптимизатора?")

    assert intent.kind == "algorithm_parameter"
    assert intent.parameter_name == "n_gen"


def test_run_optimization_extracts_alias_overrides():
    intent = parse_district_optimization_intent(
        "Оптимизируй квартал target_id=86 с размером популяции 20 и количеством поколений 50"
    )

    assert intent.kind == "run_optimization"
    assert intent.target_id == 86
    assert intent.pop_size == 20
    assert intent.n_gen == 50


def test_problem_statement_consultation_request():
    intent = parse_district_optimization_intent("Какие настройки оптимизатора есть?")

    assert intent.kind == "problem_statement"


def test_constraints_request_is_not_mapped_to_problem_statement():
    intent = parse_district_optimization_intent("Покажи ограничения в задаче оптимизации")

    assert intent.kind == "constraints"


def test_set_algorithm_parameter_request():
    intent = parse_district_optimization_intent("Измени размер популяции на 20")

    assert intent.kind == "set_algorithm_parameter"
    assert intent.pop_size == 20
    assert intent.parameter_name == "pop_size"


def test_set_algorithm_parameter_request_without_target_id_and_with_range_phrase():
    intent = parse_district_optimization_intent("Измени популяцию с 10 на 15")

    assert intent.kind == "set_algorithm_parameter"
    assert intent.pop_size == 15
    assert intent.target_id is None


def test_optimization_how_it_works_is_not_treated_as_run_request():
    intent = parse_district_optimization_intent("Как работает оптимизация кварталов?")

    assert intent.kind == "unknown"


def test_optimization_domain_detector():
    assert looks_like_district_optimization_request("Какой размер популяции?")
    assert looks_like_district_optimization_request("Измени размер популяции на 20")
    assert not looks_like_district_optimization_request("Что такое город?")
