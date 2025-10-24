# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Dict, Iterator, List, Literal, Optional, Union
import copy
from qwen_agent.agents import FnCallAgent
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, Message
from concurrent.futures import ThreadPoolExecutor, as_completed

class OpenSWEGrepAgent(FnCallAgent):
    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        max_tool_calls = kwargs.get("max_tool_calls", 100)
        max_llm_calls = kwargs.get("max_llm_calls", 100)
        max_input_tokens = kwargs.get("max_input_tokens", 128000)
        response = []
        while True and max_llm_calls > 0 and max_tool_calls > 0:
            max_llm_calls -= 1

            extra_generate_cfg = {'lang': lang, 'max_input_tokens': max_input_tokens}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            while True:
                try:
                    output_stream = self._call_llm(messages=messages,
                                                functions=[func.function for func in self.function_map.values()],
                                                extra_generate_cfg=extra_generate_cfg)
                    output: List[Message] = []
                    for output in output_stream:
                        if output:
                            yield response + output
                    break
                except:
                    time.sleep(1)
            if output:
                response.extend(output)
                messages.extend(output)
                tool_calls = []  # [(index, out, tool_name, tool_args), ...]
                for idx, out in enumerate(output):
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        if max_tool_calls == 0:
                            break
                        tool_calls.append((idx, out, tool_name, tool_args))
                        max_tool_calls -= 1

                if not tool_calls:
                    break

                tool_results = [None] * len(tool_calls)
                with ThreadPoolExecutor() as executor:
                    future_to_index = {
                        executor.submit(self._call_tool, tool_name, tool_args, messages=messages, **kwargs): i
                        for i, (_, _, tool_name, tool_args) in enumerate(tool_calls)
                    }
                    for future in as_completed(future_to_index):
                        i = future_to_index[future]
                        tool_results[i] = future.result()

                for (idx, out, tool_name, tool_args), tool_result in zip(tool_calls, tool_results):
                    fn_msg = Message(
                        role=FUNCTION,
                        name=tool_name,
                        content=tool_result,
                        extra={'function_id': out.extra.get('function_id', '1')}
                    )
                    messages.append(fn_msg)
                    response.append(fn_msg)
                    yield response

        yield response
        