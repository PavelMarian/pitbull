# D1: скилл `pit-repair` и прогон паритета. Где что лежит

Одна страница для навигации. Числа и разбор: `docs/experiments/d1-skill-repair-parity.md`.

## Итог

Скилл воспроизводит F1-цикл A2 без потери результата: CLEAN@1 = 75/76,
CLEAN@5 = 76/76 (A2: 65/76 и 75/76), Мак-Немар p = 1.0, вердикт `positive`
(статус `in_review`, шаг D1 в `docs/PLAN.md`). Стоимость прогона $0.478,
578 вызовов модели, около 70 минут при трёх параллельных сессиях.

## Скилл

| Путь | Что |
|---|---|
| `.claude/skills/pit-repair/SKILL.md` | инструкция агенту: интерфейс, детекция, текст F1 дословно, повторная проверка, до 5 итераций, спорные случаи |
| `.claude/skills/pit-repair/scripts/pit_check.py` | дифференциальная проверка (witness + canary на 6 моментах), песочница из A2, счётчик итераций в `pit_state.json`, отказ при ключе в окружении |
| `.claude/skills/pit-repair/bin/pit-check` | обёртка, интерпретатор из `$PIT_PYTHON` |
| `.claude/skills/pit-repair/references/background.md` | откуда числа и протокол (A2–A5, пилот D) |

Для OpenHarness скилл подключается симлинком в `<config>/skills/pit-repair`
(делает `pilot/d1_make_ohcfg.py`); `.claude/skills` этот харнесс не читает.

## Прогон и результаты

| Путь | Что |
|---|---|
| `pilot/d1_make_ohcfg.py` | изолированный конфиг OpenHarness `~/.openharness-d1` (ключ из `.env`, модель, симлинк скилла) |
| `pilot/d1_run.py` | раннер: 76 рабочих каталогов, промпт без имени скилла, `oh -p … --output-format stream-json` |
| `pilot/d1_score.py` | сводка по предрегистрированному правилу: CLEAN@k, Мак-Немар против A2, bootstrap, механизмы, гейты |
| `pilot/d1_results.json` | машинная сводка прогона |
| `pilot/d1_deletion_proxy.json` | прокси «починки выбрасыванием» по 76 программам |
| `pilot/d1_runs/<uid>/` | на программу: `get_features.py` (финальный код), `pit_state.json` (история итераций), `oh_stream.jsonl` (транскрипт агента), `run.json`, `prompt.txt` |
| `pilot/d1_run.log` | лог раннера |
| `docs/experiments/d1-skill-repair-parity.md` | отчёт по конвенции `docs/experiments/README.md` |
| `docs/skill-artifact-plan.md` | исходный план шага (правило §4 скопировано в отчёт до прогона) |
| `pilot/REPORT.md`, раздел «Пилот D», последний абзац | указатель на A2/A4/D1 |

## Окружение

- Проверка: `~/.venvs/pitbull-d1` (Python 3.12, pandas 2.2.3, numpy 2.2.6).
- Агент: OpenHarness 0.1.4 (`~/.openharness-venv/bin/oh`), модель
  `z-ai/glm-5.3-flash` через OpenRouter, `max_tokens` 14000; `temperature` и
  бюджет рассуждений харнесс не передаёт.
- Ключ только в `~/.openharness-d1/settings.json` (mode 600); раннер
  удаляет `*_API_KEY` из окружения OpenHarness; `pit_check.py` отказывается
  работать при ключе в окружении.

## Воспроизведение

```bash
uv venv ~/.venvs/pitbull-d1 --python 3.12
uv pip install --python ~/.venvs/pitbull-d1/bin/python "pandas==2.2.3" "numpy==2.2.6"
python3 pilot/d1_make_ohcfg.py
~/.venvs/pitbull-d1/bin/python pilot/d1_run.py
D1_PRICE_IN_PER_M=0.09 D1_PRICE_OUT_PER_M=0.30 python3 pilot/d1_score.py
```

## Черновик для статьи (не применён)

Цикл детекции и починки, провалидированный в §RQ0/RQ2, выпущен также как
переиспользуемый скилл `pit-repair` для агентных редакторов кода. На корпусе
из 76 программ (§RQ0) упаковка в скилл воспроизводит CLEAN@5 = 76/76 против
75/76 у прямого вызова (точный тест Мак-Немара, p = 1.0), что мы приводим
как практический артефакт исследования.
