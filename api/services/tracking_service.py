"""Tracking inference service over the existing GNN predictor."""


class TrackingService:
    def __init__(self, registry) -> None:
        self.registry = registry

    def predict(self, request):
        predictor = self.registry.require("gnn")
        result = predictor.predict_events(request.events)
        return {"model": "spatiotemporal_gnn", "predicted_tracks": result.get("predicted_tracks", []), "edge_predictions": result.get("edge_predictions", []), "source_mode": "synthetic_demo"}
