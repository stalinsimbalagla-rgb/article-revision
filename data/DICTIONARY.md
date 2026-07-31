# Data dictionary

The synthetic demonstration retains the source-schema column names so that the
public code follows the audited workflow.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `GRUPO` | categorical | — | Synthetic process group |
| `MOTOR` | identifier | — | Synthetic asset code |
| `Nombre` | categorical | — | Synthetic equipment description |
| `Fecha` | date | day | Event date |
| `Falla` | binary | — | Supplied failure label |
| `Mantenimiento` | binary | — | Supplied maintenance label |
| `Tiempo de Reparacion (h)` | numeric | h | Recorded intervention duration |
| `Potencia` | numeric | kW | Nominal power |
| `Voltaje (V)` | numeric | V | Nominal voltage |
| `Corriente (A)` | numeric | A | Nominal current |

`objetivo_h` is the calendar-hour interval from one supplied failure label to
the next supplied failure label for the same asset. It is not claimed to be a
confirmed physical time between failures.
