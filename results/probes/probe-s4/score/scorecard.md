# Scorecard: run probe-s4

- model: `deepseek/deepseek-v4-flash-vision-exp`
- prompt sha256: `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d`
- git: `f8403ff66eea9600c73ff80626bff9b700bdc498`
- gt hash: `10b03954d4b95f4377280f419089573c2e24bc2c895c5e9ff6ee60e235bca803`
- dataset hash: `623e4c891b2cf3db43c8d5e76c7a3f6146a12a4093ba80263d8966ad19f1a699`
- scored: 12 document(s); unscored (no label): 0
- scored at: 2026-10-08T22:36:54+00:00

## Fields

| metric | value |
| --- | --- |
| labeled | 154 |
| correct | 150 |
| accuracy (abstentions are errors) | 0.974 |
| selective accuracy | 1.000 |
| coverage | 0.974 |
| abstention rate | 0.026 |
| no prediction rate | 0.000 |

| field | labeled | correct | accuracy | selective | coverage |
| --- | --- | --- | --- | --- | --- |
| vendor_name | 12 | 12 | 1.000 | 1.000 | 1.000 |
| vendor_address | 12 | 12 | 1.000 | 1.000 | 1.000 |
| vendor_tax_id | 12 | 12 | 1.000 | 1.000 | 1.000 |
| buyer_name | 12 | 12 | 1.000 | 1.000 | 1.000 |
| document_number | 12 | 12 | 1.000 | 1.000 | 1.000 |
| document_date | 12 | 12 | 1.000 | 1.000 | 1.000 |
| due_date | 12 | 12 | 1.000 | 1.000 | 1.000 |
| currency | 10 | 10 | 1.000 | 1.000 | 1.000 |
| payment_method | 12 | 12 | 1.000 | 1.000 | 1.000 |
| subtotal | 12 | 11 | 0.917 | 1.000 | 0.917 |
| tax | 12 | 11 | 0.917 | 1.000 | 0.917 |
| discount | 12 | 11 | 0.917 | 1.000 | 0.917 |
| total | 12 | 11 | 0.917 | 1.000 | 0.917 |

## Rows

- matched 516, missed 0, spurious 0, perfect 516
- precision 1.000, recall 1.000, F1 1.000, row accuracy 1.000, cell accuracy 1.000

## Errors

- abstained_on_absent: 4

## Slice: split

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| test | 154 | 0.974 | 1.000 | 0.974 |

## Slice: doc_type

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| invoice | 154 | 0.974 | 1.000 | 0.974 |

## Slice: capture

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| constructed | 154 | 0.974 | 1.000 | 0.974 |

## Cost and latency

Documents attempted (ok or failed); latency is over live calls only, nearest-rank.

| scope | docs | live | cached | paid | at list price | per doc (list) | tokens in/out | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 12 | 11 | 1 | $0.1080 | $0.1162 | $0.0097 | 68736/79661 | 21.2 s | 30.0 s |
| split=test | 12 | 11 | 1 | $0.1080 | $0.1162 | $0.0097 | 68736/79661 | 21.2 s | 30.0 s |
| doc_type=invoice | 12 | 11 | 1 | $0.1080 | $0.1162 | $0.0097 | 68736/79661 | 21.2 s | 30.0 s |
| capture=constructed | 12 | 11 | 1 | $0.1080 | $0.1162 | $0.0097 | 68736/79661 | 21.2 s | 30.0 s |

Unscored documents (no ground truth): none
