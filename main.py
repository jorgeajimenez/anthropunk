import simple_icd_10_cm as cm
import os
from anthropic import Anthropic

icd_sample = 'E08'
icd_desc = cm.get_full_data(icd_sample)

api_key = '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'

os.environ['ANTHROPIC_API_KEY'] = api_key

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "A patient has ICD code E0800. What treatment might they need?",
        }
    ],
    model="claude-3-opus-20240229",
)
print(message.content)
