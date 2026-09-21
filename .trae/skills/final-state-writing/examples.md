# Examples

## 1. Model choice

The draft says it uses GAN. The user corrects it: use only L1+LPIPS.

Preferred final text: `We use L1+LPIPS.`

## 2. Removed cache

An intermediate implementation added a cache, then the user asked to remove it.

Preferred final PR description: describe the implementation without mentioning
the cache, unless “no cache” is itself a final requirement.

## 3. Removed unsupported claim

After deleting an unsupported claim, do not add a disclaimer emphasizing that
the work does not claim it.

## 4. Decision memo

If the user explicitly requests a decision memo explaining why option B was
rejected, retain option B and the reason for rejecting it: the history is part
of the requested artifact.

## 5. Tomato and eggs

```text
Bad:
Tomato and Eggs (without Dongpo Pork)

Preferred:
Tomato and Eggs
```
