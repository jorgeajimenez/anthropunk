from loaders import TextLoader
import os
import shutil

import numpy as np
import simple_icd_10_cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
from anthropic import Anthropic
from markdown_pdf import MarkdownPdf
from markdown_pdf import Section
from tqdm import tqdm

# load the dataset, split into input (X) and output (y) variables
# dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# X = dataset[:, 0:8]
# y = dataset[:, 8]

dataset = np.loadtxt('diabetes_predictor_set.csv', delimiter=',')

predict_hd = False

if predict_hd:
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
else:
    X = dataset[:, 1:]
    y = dataset[:, 0]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


class HealthClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(21, 16)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 10)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(10, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


model = HealthClassifier()
print(model)

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 1000

step_loss = []

epoch_queue = tqdm(range(n_epochs))

for epoch in epoch_queue:
    batch_loss = 100
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i + batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        batch_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    step_loss.append(batch_loss)

# plt.plot(step_loss)
# plt.show()
# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
tp = 0
tn = 0
fp = 0
fn = 0

predictions = (model(X) > 0.20).int()

picked_single = False
single = []

for i in range(len(X)):
    # print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
    prediction = predictions[i].item()
    expected = y[i].item()

    if expected == 1:
        if expected == prediction:
            tp += 1
            if not picked_single:
                single = X[i].tolist()
                picked_single = True
        else:
            fn += 1
    else:
        if expected == prediction:
            tn += 1
        else:
            fp += 1

age_key = {
    1: [18, 24],
    2: [25, 29],
    3: [30, 34],
    4: [35, 39],
    5: [40, 44],
    6: [45, 49],
    7: [50, 54],
    8: [55, 59],
    9: [60, 64],
    10: [65, 69],
    11: [70, 74],
    12: [75, 79],
    13: [80, 130]
}
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f'{"Heart Disease Performance" if predict_hd else "Diabetes Performance"}')
print(f'Sensitivity/Specificity: {sensitivity}/{specificity}')

reference_icd = 'I21' if predict_hd else 'E08'
has_history = True
age_info = age_key[int(single[17])]
prompt = (f'The patient is between {age_info[0]} and {age_info[1]} years old '
          f'and they are suspected of {"heart disease" if predict_hd else "diabetes"}. '
          f'They {"smoke" if int(single[4]) else "do not smoke"} and have a BMI of {single[3]}. '
          f'They {"have" if has_history else "does not have"} a family history of '
          f'{"heart disease" if predict_hd else "diabetes"}. They have {"elevated" if single[0] else "normal"} blood '
          f'pressure at the moment, and {"exercise regularly" if single[6] else "have a sedentary lifestyle"}. For '
          f'billing reference, their ICD code is {reference_icd}. '
          f'What kind of things should this person do to '
          f'improve their longevity and quality of life? '
          f'Also make a good prediction about how many more years they can expect to live with this condition untreated.'
          f'Make a second prediction of how many years they can add to their lives if they make the suggested changes.'
          f'Then, give a range for what their age expectancy might be. Finally, say something really encouraging '
          f'and motivational to make the person feel good about themselves and make the changes.')

print(f'\n\nPROMPT: {prompt}\n\n')
# print(f'{predictions[i].item()}, {y[i].item()}')

icd_sample = 'E08'
icd_desc = cm.get_full_data(icd_sample)

api_key = '{ANTHROPIC_API_KEY}'

os.environ['ANTHROPIC_API_KEY'] = api_key

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)

loader = TextLoader()
loader.start()

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="claude-3-opus-20240229",
)
loader.stop()
print(f'\n\nRESPONSE: {message.content}\n\n')

if os.path.isfile('output_report_sample.md'):
    os.remove('output_report_sample.md')

shutil.copy('anthro_report.md', f'output_report_sample.md')
with open('anthro_report.md', 'r') as report:
    template_contents = report.read()

report_content = template_contents.replace("{{report_contents_body}}", str(message.content[0].text))
report_content = report_content.replace("{{report_prompt_used}}", prompt)
with open('output_report_sample.md', 'w+') as report:
    report.write(report_content)

pdf = MarkdownPdf(toc_level=2)

pdf.add_section(Section(report_content, toc=False))
pdf.meta['title'] = 'Health Report'

pdf.save('output_report_sample.pdf')
