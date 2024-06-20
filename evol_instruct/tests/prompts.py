import json
from pyprojroot import here


prompts = {
  'in_depth_evolving': {
    'base': """<human>: I want you to act as a prompt rewriter.
Your objective is to rewrite the #Given Prompt# into a more complex version.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the context in #Given Prompt#.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
You SHOULD complicate the given prompt {operation}
#Given Prompt#:
{instruction}
<bot>: #Rewritten Prompt#:""",
    'operations': {
      'add-constraints': "by adding one more constraints/requirements into #Given Prompt#",
      'deepening': "if #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.",
      'concretizing': "by replacing general concepts with more specific concepts.",
      'increase-reasoning-steps': "if #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    }
  },
  'in_breadth_evolving': """<human>: I want you to act as a prompt creator.
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.
The #Created Prompt# must be reasonable and must be understood and responded by humans.
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#.
Your response only contains the #Created Prompt# and no explanation of the new prompt. Do not provide a response to either the #Given Prompt# or the #Created Prompt#.
#Given Prompt#:
{instruction}
<bot>: #Created Prompt#:""",
  'equality_check_prompt': """Do you think the following two instructions are equal to each other in that they meet the following requirements:
1. They have same constraints and requirements.
2. They have same depth and breadth of the inquiry.
The First Prompt: {original_instruction}
The Second Prompt: {evolved_instruction}
Your response should be one of either 'equal' or 'not equal'."""
}

with open(here('evol_instruct/prompts.json'), 'w') as jsonfile:
  json.dump(prompts, jsonfile)