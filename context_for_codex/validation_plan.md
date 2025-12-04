### Validation plan

Phase 1 validation goal:
- Greenfield’s first required milestone is:
  - **Demonstrate a model that successfully predicts a real failure event that already happened**, using only historical data **where that specific failure was not seen in training**.
- In practice:
  - Hold out at least one known Priority 1 service request (per problem asset, if possible) as a **true test case**.
  - Train the asset’s model on earlier history and earlier failures only.
  - Show that, when run on the historical data leading up to the held-out failure, the model:
    - Produces a **high failure probability** in the final days before the event.
    - Clearly demonstrates predictive capability (not just post-hoc detection).
- This “predict one known failure we didn’t train on” demonstration is the **key success criterion for Phase 1** and will be used to justify further deployment and expansion.
