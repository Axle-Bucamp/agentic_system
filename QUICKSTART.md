# Guide de Démarrage Rapide

Ce guide vous permet de démarrer le système de trading agentique en quelques minutes.

## Prérequis

- **Docker** et **Docker Compose** installés
- Au moins **4 GB de RAM** disponible
- **Ports disponibles** : 8000, 8001, 3000, 5432, 6379, 9090

## Installation en 3 Étapes

### 1. Configuration

Copiez le fichier d'environnement et configurez vos clés API :

```bash
cp .env.example .env
nano .env  # ou utilisez votre éditeur préféré
```

**Variables importantes à configurer :**

```env
# API MCP pour les prédictions DQN
MCP_API_URL=https://forecasting.guidry-cloud.com

# OpenAI API pour l'analyse de news (optionnel)
OPENAI_API_KEY=your_openai_key_here

# VLLM endpoint pour LLM local (optionnel, alternative à OpenAI)
VLLM_ENDPOINT=http://your-vllm-server:8000/v1

# RPC URLs pour les blockchains (pour le copy trading)
ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your_key
BSC_RPC_URL=https://bsc-dataseed.binance.org/
```

### 2. Démarrage

Lancez tous les services avec le script de démarrage :

```bash
./start.sh
```

Ou manuellement avec Docker Compose :

```bash
docker-compose up --build -d
```

### 3. Vérification

Vérifiez que tous les services sont opérationnels :

```bash
docker-compose ps
```

Vous devriez voir tous les services avec le statut `Up`.

## Accès aux Interfaces

Une fois les services démarrés, vous pouvez accéder à :

| Service | URL | Description |
| :--- | :--- | :--- |
| **API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Documentation Swagger de l'API |
| **API Health** | [http://localhost:8000/health](http://localhost:8000/health) | Vérification de l'état du système |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Dashboards de monitoring |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Métriques système |
| **DEX Simulator** | [http://localhost:8001](http://localhost:8001) | Interface du simulateur DEX |

**Identifiants Grafana par défaut :**
- Username: `admin`
- Password: `admin_change_in_prod`

## Test Rapide

### Via l'API

Vérifiez l'état du portfolio :

```bash
curl http://localhost:8000/api/portfolio
```

Obtenez les performances :

```bash
curl http://localhost:8000/api/performance
```

Consultez les signaux pour BTC :

```bash
curl http://localhost:8000/api/signals/BTC
```

### Via le Script de Test

Exécutez les tests manuels pour valider chaque agent :

```bash
python3.11 test_manual.py
```

## Commandes Utiles

### Voir les logs

```bash
# Tous les services
docker-compose logs -f

# Un service spécifique
docker-compose logs -f orchestrator-agent
docker-compose logs -f dqn-agent
```

### Redémarrer un agent

```bash
docker-compose restart dqn-agent
```

### Arrêter le système

```bash
docker-compose down
```

### Arrêter et supprimer les volumes

```bash
docker-compose down -v
```

## Architecture des Agents

Le système est composé de 7 agents indépendants :

| Agent | Cycle | Rôle |
| :--- | :--- | :--- |
| **Orchestrator** | 5 min | Coordonne tous les agents et prend les décisions finales |
| **DQN Agent** | 5 min | Obtient les prédictions de l'API MCP |
| **Chart Agent** | 5 min | Analyse technique (RSI, MACD, Bollinger) |
| **Risk Agent** | 2 min | Évalue les risques et valide les trades |
| **Memory Agent** | 10 min | Maintient l'historique et calcule les métriques |
| **News Agent** | 15 min | Analyse le sentiment des actualités crypto |
| **CopyTrade Agent** | 3 min | Surveille les transactions on-chain |

## Validation Humaine

Pour les trades importants ou risqués, le système peut demander une validation humaine.

### Consulter les validations en attente

```bash
curl http://localhost:8000/api/validations/pending
```

### Approuver un trade

```bash
curl -X POST http://localhost:8000/api/validations/{request_id}/respond \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "xxx",
    "approved": true,
    "feedback": "Approved after manual review"
  }'
```

## Ajouter un Wallet à Suivre

Pour activer le copy trading d'un wallet performant :

```bash
curl -X POST http://localhost:8000/api/wallets/track \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0x...",
    "blockchain": "ETH"
  }'
```

## Dépannage

### Les agents ne démarrent pas

Vérifiez les logs pour identifier l'erreur :

```bash
docker-compose logs orchestrator-agent
```

### Erreur de connexion Redis

Vérifiez que Redis est bien démarré :

```bash
docker-compose ps redis
docker-compose logs redis
```

### Erreur de connexion PostgreSQL

Vérifiez que PostgreSQL est bien démarré et initialisé :

```bash
docker-compose ps postgres
docker-compose logs postgres
```

### L'API MCP ne répond pas

Vérifiez votre `MCP_API_URL` dans le fichier `.env` et assurez-vous que l'API est accessible.

## Prochaines Étapes

1. **Configurez Grafana** : Importez les dashboards pré-configurés pour visualiser les performances
2. **Ajustez les paramètres** : Modifiez `core/config.py` selon vos besoins (capital, limites, intervalles)
3. **Ajoutez des wallets** : Configurez le copy trading avec des wallets performants
4. **Activez les alertes** : Configurez les notifications pour les événements importants
5. **Passez en production** : Déployez sur Docker Swarm ou Kubernetes

## Support

Pour toute question ou problème, consultez :

- **Documentation complète** : `README.md`
- **Architecture détaillée** : `system_analysis.md`
- **Tests** : `tests/` et `test_manual.py`

