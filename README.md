# Agentic Trading System

Ce projet est un système de trading multi-agents de grade AAA, conçu pour être modulaire, scalable et performant. Il exploite des prédictions de modèles DQN, des analyses techniques, des analyses de sentiment, et des stratégies de gestion des risques pour prendre des décisions de trading automatisées.

## Architecture du Système

Le système est construit autour d'une architecture microservices, où chaque agent est un service indépendant communiquant via un bus de messages (Redis Pub/Sub). L'ensemble est orchestré par Docker Compose pour un déploiement local facile et est prêt pour une migration vers Docker Swarm ou Kubernetes.

### Composants Principaux

- **API Principale (FastAPI)** : Point d'entrée pour l'interaction avec le système, la consultation des données et la validation humaine.
- **Agents Spécialisés (Python)** : 7 agents indépendants, chacun avec une responsabilité unique.
- **Base de Données (PostgreSQL)** : Pour la persistance des trades, des signaux et des métriques de performance.
- **Cache & Message Bus (Redis)** : Pour l'état partagé, la communication inter-agents et le caching.
- **Simulateur DEX** : Une version adaptée du simulateur fourni pour les tests en environnement contrôlé.
- **Monitoring (Prometheus & Grafana)** : Pour la surveillance en temps réel des performances du système.

### Agents

| Agent | Rôle | Technologies | Cycle |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | Coordonne tous les agents, agrège les signaux et prend les décisions finales. | AutoGen, LangChain | 5 min |
| **DQN Agent** | Interface avec l'API MCP pour obtenir les prédictions (BUY/SELL/HOLD). | HTTPX | 5 min |
| **Chart Agent** | Effectue l'analyse technique (RSI, MACD, Bollinger Bands). | Pandas, TA | 5 min |
| **Risk Agent** | Évalue les risques, valide les décisions et gère les limites de position. | NumPy | 2 min |
| **Memory Agent** | Maintient l'historique des trades et calcule les métriques de performance. | Redis, SQL | 10 min |
| **News Agent** | Analyse le sentiment des actualités crypto en utilisant un LLM (VLLM/OpenAI). | LangChain, OpenAI | 15 min |
| **CopyTrade Agent** | Surveille les transactions on-chain de wallets performants. | Web3.py, Etherscan API | 3 min |

## Fonctionnalités

- **Architecture Multi-Agents** : Conception modulaire et scalable avec AutoGen et LangChain.
- **Pipeline de Décision Hybride** : Combine des signaux de ML, d'analyse technique, de sentiment et de risque.
- **Gestion des Risques Avancée** : Limites de position par tier d'actifs, stop-loss dynamiques (via Risk Agent), et validation des trades.
- **Trading Multi-Intervalle** : Observe les marchés en minutes, prend des décisions en heures, et utilise des prévisions journalières.
- **Validation Humaine Optionnelle** : Les trades importants ou risqués peuvent nécessiter une approbation manuelle via l'API.
- **Monitoring Complet** : Intégration avec Prometheus et Grafana pour des dashboards de performance.
- **CI/CD Robuste** : Pipeline GitHub Actions pour les tests, le linting, la couverture de code, la construction Docker et l'analyse de sécurité.
- **API Complète** : Endpoints FastAPI pour suivre le portfolio, les performances, les signaux, et interagir avec le système.

## Installation et Démarrage

### Prérequis

- Docker et Docker Compose
- Un fichier `.env` basé sur `.env.example` avec vos clés API.

### Démarrage Rapide

1.  **Clonez le projet** :

    ```bash
    git clone <url-du-repo>
    cd agentic-trading-system
    ```

2.  **Configurez les variables d'environnement** :

    Copiez `.env.example` vers `.env` et remplissez les clés API nécessaires (au minimum `OPENAI_API_KEY` si vous utilisez l'analyse de news).

    ```bash
    cp .env.example .env
    ```

3.  **Lancez le système avec Docker Compose** :

    ```bash
    docker-compose up --build -d
    ```

4.  **Vérifiez que tous les services sont en cours d'exécution** :

    ```bash
    docker-compose ps
    ```

    Vous devriez voir tous les services (redis, postgres, trading-api, et tous les agents) avec le statut `Up`.

5.  **Accédez à l'API et à la documentation** :

    - **API Docs (Swagger)** : [http://localhost:8000/docs](http://localhost:8000/docs)
    - **Grafana Dashboards** : [http://localhost:3000](http://localhost:3000) (login: admin / admin_change_in_prod)
    - **Prometheus** : [http://localhost:9090](http://localhost:9090)

## API Endpoints

L'API FastAPI expose de nombreux endpoints pour interagir avec le système. Voici quelques exemples :

- `GET /api/portfolio` : Récupère l'état actuel du portfolio.
- `GET /api/performance` : Obtient les métriques de performance (win rate, PnL, Sharpe ratio).
- `GET /api/signals/{ticker}` : Affiche les derniers signaux de tous les agents pour un ticker donné.
- `GET /api/validations/pending` : Liste les trades en attente de validation humaine.
- `POST /api/validations/{request_id}/respond` : Approuve ou rejette un trade.
- `GET /api/history/trades` : Récupère l'historique des trades.

Consultez la documentation Swagger sur [http://localhost:8000/docs](http://localhost:8000/docs) pour la liste complète.

## Tests et Qualité du Code

Le projet est configuré avec une suite de tests complète utilisant `pytest`.

- **Lancer les tests** :

  ```bash
  pytest
  ```

- **Rapport de couverture** : Un rapport de couverture est généré dans `htmlcov/index.html`.

La pipeline CI/CD sur GitHub Actions assure que chaque commit est testé, linté et analysé pour les vulnérabilités de sécurité.

## Déploiement en Production

Le système inclut maintenant une configuration de production complète avec observabilité, sécurité et optimisations de performance.

### Configuration de Production

- **Docker Compose Production** : `docker-compose.prod.yml` avec réplication, limites de ressources et sécurité renforcée
- **Nginx Reverse Proxy** : Load balancing, rate limiting et compression
- **Monitoring Avancé** : Prometheus, Grafana et Jaeger pour l'observabilité complète
- **Sécurité** : JWT, rate limiting, headers de sécurité et chiffrement

### Démarrage Rapide Production

1. **Configuration** :
   ```bash
   cp config/production.env.example .env.prod
   # Éditez .env.prod avec vos valeurs
   ```

2. **Déploiement** :
   ```bash
   # Windows
   scripts/deploy.bat
   
   # Linux/Mac
   chmod +x scripts/deploy.sh
   ./scripts/deploy.sh
   ```

3. **Services disponibles** :
   - **API Trading** : http://localhost:8000
   - **Grafana** : http://localhost:3000
   - **Prometheus** : http://localhost:9090
   - **Jaeger** : http://localhost:16686

### Fonctionnalités de Production

- **Observabilité** : Métriques, logs structurés et tracing distribué
- **Sécurité** : Authentification JWT, rate limiting et validation d'entrée
- **Performance** : Connection pooling, caching Redis et traitement asynchrone
- **Scalabilité** : Réplication automatique et load balancing
- **Monitoring** : Dashboards Grafana et alertes Prometheus

Consultez [DEPLOYMENT.md](DEPLOYMENT.md) pour un guide complet de déploiement en production.

## Auteurs

- **Manus AI**

