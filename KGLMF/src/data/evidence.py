"""Pack public HotpotQA paragraphs within the fixed evidence budget."""

def evidence_text(context, tokenizer, budget=1800):
    if isinstance(context, dict):
        context = list(zip(context['title'], context['sentences']))
    if not isinstance(context, list) or not 1 <= len(context) <= 10:
        raise ValueError('Hotpot distractor requires its supplied public paragraphs (up to ten)')
    blocks = ['['+str(title)+']\n'+''.join(sentences) for title,sentences in context]
    tokens = [tokenizer.encode(block, add_special_tokens=False) for block in blocks]
    # Water-fill equally, then return spare capacity to paragraphs that need it.
    kept = [0]*len(tokens)
    remaining = budget
    while remaining and any(n < len(t) for n,t in zip(kept,tokens)):
        for i,t in enumerate(tokens):
            if remaining and kept[i] < len(t):
                kept[i] += 1
                remaining -= 1
    text = '\n\n'.join(tokenizer.decode(t[:n], skip_special_tokens=True) for t,n in zip(tokens,kept))
    return text, dict(paragraphs=len(context), original_tokens=sum(map(len,tokens)), retained_tokens=sum(kept),
                      truncated_paragraphs=sum(n<len(t) for n,t in zip(kept,tokens)), allocation=kept,
                      rule='equal_share_with_spare_redistribution_in_original_order', token_budget=budget)
