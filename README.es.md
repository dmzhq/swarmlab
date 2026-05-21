<!-- IDLE-COMPUTE · readme_translation_es · 2026-05-18T13:23:39.990Z -->
# swarmlab

> Ejecuciones multi-agente determinísticas y reproducibles con cualquier proveedor de LLM.

[![CI](https://github.com/dmzhq/swarmlab/actions/workflows/ci.yml/badge.svg)](https://github.com/dmzhq/swarmlab/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)

> [!NOTE]
> Desarrollo temprano. La versión 0.1 incluirá el motor de *replay* determinístico
> y paridad de proveedores para OpenAI, Anthropic, y modelos servidos localmente vía
> vLLM y Ollama. Estrellen el repo para seguir el progreso.

---

## Por qué

Depurar sistemas multi-agente en 2026 se resume en "volver a correrlo y esperar que haga lo mismo". Las herramientas de trazado de proveedores te muestran *qué* pasó, pero no te permiten reejecutar una rama de un rastro pasado sin gastar dinero de nuevo.

`swarmlab` hace que las ejecuciones multi-agente sean **determinísticas por construcción** (muestreo con semillas, llamadas a herramientas registradas, caché con direccionamiento por contenido) y **completamente reproducible** desde un único `run-id`. Compila un DAG de agentes y herramientas (en YAML o Python) en un plan de ejecución, captura cada llamada a herramienta y cada respuesta de LLM en un almacén direccionado por contenido, y te permite reproducir cualquier rama de una ejecución pasada con **cero costo de LLM**.

Es **agnóstico al proveedor** — OpenAI, Anthropic, Mistral, y modelos locales servidos con vLLM — y viene con un *eval harness* integrado para que puedas medir la deriva de calidad entre proveedores en el mismo rastro.

## Estado

| Componente | Estado |
|------------------------------------|---------------|
| Toolchain bootstrap (uv, ruff, pyright, pytest) | ✅ enviado |
| Esquema DAG + cargadores | 🚧 en progreso |
| Almacén direccionado por contenido | 🚧 en progreso |
| Adaptador de proveedores + semillas determinísticas | planificado |
| Planificador con reintentos + *fan-out* | planificado |
| Motor de *replay* | planificado |
| *Eval harness* | planificado |
| Exportación OpenTelemetry | planificado |

## Instalar

```bash
# próximamente (se publicará en PyPI con 0.1)
pip install swarmlab
```

## Primer uso (Quickstart)

```bash
# planificado para 0.1 — marcador de posición
swarmlab run examples/01_minimal_two_agents.yaml
swarmlab replay <run-id>
```

## Principios de diseño

1. **Determinismo sobre velocidad.** Misma entrada + misma semilla = salida idéntica a nivel de byte.
2. **Portabilidad de proveedores.** Cambia OpenAI por Anthropic para una ejecución local con Qwen sin reescribir el código de tus agentes.
3. **Almacenamiento direccionado por contenido.** Cada llamada a herramienta y cada respuesta de LLM son claves por el hash de sus entradas. El *replay* lee de la caché. Las llamadas reales a LLM solo ocurren en caso de fallo de caché (*cache miss*).
4. **Listo para auditoría por defecto.** Cada ejecución produce un único rastro completo que puedes entregar a un revisor, un oficial de cumplimiento o un tú futuro.

## Comparación

| | `swarmlab` | LangGraph | CrewAI | Inngest / Temporal |
|---|---|---|---|---|
| Agnóstico al proveedor | ✅ | ⚠️ (Acoplado a LangChain) | ✅ | n/a |
| Determinístico por construcción | ✅ | ❌ | ❌ | ⚠️ (Duradero, no determinístico) |
| Reproducir rama con cero costo de LLM | ✅ | ❌ | ❌ | ❌ |
| *Eval harness* integrado | ✅ | ❌ | ❌ | ❌ |
| Ejecución duradera genérica | parcial | parcial | ❌ | ✅ |

## Seguridad

Consulta [SECURITY.md](SECURITY.md) para divulgación responsable.

## Licencia

Apache-2.0 — ver [LICENSE](LICENSE).