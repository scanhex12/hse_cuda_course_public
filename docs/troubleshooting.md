# Решение распространённых проблем

## Не получается сдать задачу (submit.py падает)

Запустите скрипт с флагом **-v** (verbose), чтобы увидеть выполняемые команды и место ошибки:

```bash
python3 ../submit.py -v
```

## `fatal: No such remote 'student'`

Скрипт отправки пушит в remote с именем `student`. Проверьте:

```bash
git remote -v
```

Должен быть вывод вида:
```
origin  <URL репозитория курса>
student <URL вашего личного репозитория>
```

Если `student` нет, добавьте:
```bash
git remote add student <URL вашего личного репо>
```

URL берите со страницы вашего репозитория (Clone → Clone with SSH).

## Permission denied (publickey) при push

SSH-ключ не настроен или не добавлен в GitLab. Добавьте публичный ключ в настройках GitLab (Settings → SSH Keys) и убедитесь, что `ssh -T git@<ваш-gitlab-host>` успешно подключается.

## Ошибка «Task config not found»

Скрипт ищет `task.json` или `.tester.json` в директории задачи. Запускайте `submit.py` из директории задачи (например, `00-hello-world`) или передайте путь: `python3 submit.py 00-hello-world`.

## CI падает при изменении «запрещённых» файлов

В каждой задаче в `task.json` указано поле `allow_to_change` — список файлов, которые можно менять. Не изменяйте тесты и другие файлы вне этого списка.
