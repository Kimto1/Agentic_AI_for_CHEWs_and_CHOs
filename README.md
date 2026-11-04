# Community Health Guideline Navigator

A local demonstration prototype for Community Health Extension Workers (CHEWs) and Community Health Officers (CHOs). It searches Nigeria's 2024 National Standing Orders and uses a locally running Ollama model to draft a response from retrieved manual pages.

## Workflow

```mermaid
flowchart TD
    A["Worker selects cadre and enters a question"] --> B["Search the matching Standing Orders"]
    B --> C{"Relevant pages found?"}
    C -- "No" --> D["Say no relevant guidance was found"]
    C -- "Yes" --> E["Send retrieved text to local Ollama model"]
    E --> F["Show AI draft and retrieved filename and PDF pages"]
    F --> G["Health worker checks the cited manual pages"]

## Project QR code

Scan this code to open the GitHub repository.

![QR code linking to the project repository](community_health_guideline_navigator_qr.png)
