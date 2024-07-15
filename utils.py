
def apply_chat_formatting(
    tokenizer,
    prompt,
    def_completion,
    adv_completion,
    use_tokenizer_template,
    system_prompt,
    custom_prompt_template,
    custom_completion_template,
    add_generation_prompt=True
):
    if use_tokenizer_template:
        if system_prompt is not None:
            prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            prompt_messages = [{"role": "user", "content": prompt}]
        
        prompt_str = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=add_generation_prompt, tokenize=False)
    else:
        if system_prompt is not None:
            prompt_str = custom_prompt_template.format(system_prompt=system_prompt, prompt=prompt)
        else:
            prompt_str = prompt
    
    if custom_completion_template is not None:
        adv_str = custom_completion_template.format(completion=adv_completion)
        def_str = custom_completion_template.format(completion=def_completion)
    else:
        adv_str = adv_completion
        def_str = def_completion
    
    return prompt_str, adv_str, def_str
