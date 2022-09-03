### Тестовое задание на позицию Junior Data Scientist в bewise.ai

Cкрипт для парсинга диалогов менеджеров и клиентов. 

Главные задачи, которые должен выполнять скрипт:
 - Извлекать реплики с приветствием – где менеджер поздоровался. 
 - Извлекать реплики, где менеджер представил себя. 
 - Извлекать имя менеджера. 
 - Извлекать название компании. 
 - Извлекать реплики, где менеджер попрощался.
 - Проверять требование к менеджеру: «В каждом диалоге обязательно необходимо поздороваться и попрощаться с клиентом»

**Запуск:**

python dialogs_parser.py path_to_data path_to_save path_to_natasha

path_to_data - путь к папке с исходным файлом 'test_data.csv'
path_to_save - путь для сохранения результатов
path_to_natasha - путь к папке с библиотекой natasha (для загрузки необходимых модулей)

**Пример запуска:**

python dialog_parsing.py C:\Users\user\Desktop\Bewise C:\Users\user\Desktop\Bewise F:\\Data_Science\\Anaconda\\Lib\\site-packages\\natasha

**Результат работы:**

В папку path_to_save  сохраняются два файла: 
 - test_data_with_results.csv - исходный файл с добавленными результатами парсинга
 - summary_results.csv - сводные результаты по каждому диалогу, содержащие следующую информацию:
   - dlg_id - id диалога
   - greeting_phrase - фраза, в которой менеджер приветствует клиента,
   - presentation_phrase - фраза, в которой менеджер представляется,
   - farewell_phrase - фраза, в которой менеджер прощается с клиентом,
   - manager_name - имя менеджера,
   - company - название компании,
   - is_greeting - флаг, поздоровался ли менеджер,
   - is_farewell - флаг, попрощался ли менеджер,
   - correctness - общая корректность диалога (наличие приветствия и прощания)
