
# Instrucciones
El área de marketing nos invita a una de sus reuniones internas donde exponen entre otras cosas que el área de retención de la compañía ha detectado un aumento continuado en los últimos meses de las bajas de los clientes de seguro de autos y para mitigarlo quieren hacer una campaña donde se le envía un incentivo a los clientes que vayan a renovar, pero el presupuesto es limitado.

Para ayudarles a determinar a qué clientes merece la pena retener, les proponemos hacer un modelo predictivo.


## Ejercicio 1: Fuga de clientes

Para este ejercicio, como integrante del área de Advanced Analytics tienes que analizar la BBDD para comprender los factores que afectan a la fuga de clientes. El objetivo es definir el problema, llevar a cabo tareas de:
- data wrangling
- ingeniería de variables
- análisis exploratorio 

Proporcionando ideas y conclusiones y por último explorar diferentes técnicas de modelización, eligiendo la mejor y comentando resultados.

**La variable dependiente es: "anula"**
Adicionalmente se cuenta con otra BBDD con variables sociodemográficas por zona más relevantes para enriquecer el modelo.

Se espera que el código a presentar sea lo más simple y limpio posible, bien comentado y explicando el razonamiento de cada paso y sus conclusiones.

El lenguaje a usar deberá ser Python o R sobre un notebook.

Además, para presentárselo al área de marketing, nos piden explicar las siguientes cuestiones: (para lo cual reserva una parte de la BBDD)

- Interpreta los valores SHAP de los 3 principales factores más importantes, asi como sus interacciones.
- Explica que métricas has usado para determinar que el modelo es apropiado
- Que punto de corte has elegido y explica porque crees que sería el mas apropiado para este caso
- Presentar diferentes propuestas de a cuantos, y a que clientes enviarías el incentivo de 50 euros y las ventajas/desventajas de cada una, así como el gasto esperado en incentivos, razonando las respuestas. 
- Que perfiles de clientes es el que más riesgo tiene de fuga? Y cuales menos?


## Ejercicio 2: Desbalanceo, Ensamble, post-proceso

Se pide mejorar el modelo para incrementar la tasa de acierto, para ello se porpone lo siguiente:

- Aplica la técnica más apropiada para intentar mejorar el desbalanceo en el modelo anterior
- Después realiza un modelo de ensamblaje  que consideres más apropiado
- Aplica las técnicas adicionales que consideres que incrementen más la precisión del modelo.
- Cual es la probabilidad media real y cual es la estimada antes y despues de aplicar las mejoras? 
- Aplica la técnica que consideres apropiada para asemejar la probabilidad media real y la estimada
- Por ultimo, contruye la variable beneficio= prima-importe de siniestros y vuelve a presentar diferentes propuestas de a que clientes y cuantos enviarias el incentivo. Y si cambiarias el importe del incentivo.


## Ejercicio 3:

Se proporciona una BBDD con la descripción de los siniestros de los clientes, tipificados por causa.

El objetivo del ejercicio es usar el modelo de IA que consideres más apropiado y sobre el, hacer un fine tuning para conseguir que tipifique las causas a partir de la descripción.


# Configuracion

- uv



# Schema en git

.
├── main.py
├── notebooks/
│   └── ejercicio1.ipynb
├── src/
│   └── utils/
│       ├── __init__.py
│       ├── text.py              # funciones genéricas (quitar tildes, slug)
│       └── data_cleaning.py     # funciones domain-specific + transformers
├── config/
│   └── canonicos.yaml           # mapeos por columna (editable)
├── pyproject.toml
└── uv.lock