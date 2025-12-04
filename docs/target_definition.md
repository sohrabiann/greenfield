# Label Definition

Failure labels are anchored on Priority 1 service requests for Dryer Area problem assets.

- **Failure anchor:** the reported timestamp of each Priority 1 service request.
- **Horizon:** 4 days after each failure anchor.
- **Failure window:** any DCS timestamp that occurs within 4 days **before** a Priority 1 request is marked `failure_within_4d`.
- **Healthy window:** timestamps with no Priority 1 request in the next 4 days are marked `healthy`.

Healthy periods deliberately exclude future-failure timestamps to avoid leakage.
