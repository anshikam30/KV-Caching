import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
import os
from collections import OrderedDict
import matplotlib.pyplot as plt 

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # TODO: Implement single-head attention
        # Create query, key, value, and output projection layers
        # Remember to set bias=False for query, key, value projections. 
        # Example : self.W_q = nn.Linear(d_model, d_model, bias=False)

        # Creating query, key, value, and output projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=False)            #Query Layer
        self.W_k = nn.Linear(d_model, d_model, bias=False)            #Key Layer
        self.W_v = nn.Linear(d_model, d_model, bias=False)            #Value Layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)            #Output Layer

    def forward(self, x, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for single-head attention
        # 1. Compute query, key, value projections
        # 2. Handle KV caching if use_cache=True
        # 3. Compute attention scores with scaling #hint use torch.bmm
        # 4. Apply causal masking #hint: use torch.triu
        # 5. Apply softmax to get attention weights
        # 6. Compute context vector #hint: use torch.bmm
        # 7. Compute output and return
        # Your code here

        B, T, d = x.size()  #Batch, SeqLen, d_model

        #1. Compute query, key, value projections
        q = self.W_q(x)                                # (B,T, d_model)
        k = self.W_k(x)                                # (B,T, d_model)
        v = self.W_v(x)                                # (B,T, d_model)

        #2. Handling KV caching if use_cache=True
        #If we have past_kv, append new K,V to the old ones along sequence dimension
        #so that the old tokens plus the new tokens are all available.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)          # Append current keys to cached ones
            v = torch.cat([past_v, v], dim=1)          # Concatenate along the seq_len dimension

        # new_past_kv is the updated (k, v) we want to feed to the next forward call
        # but only if use_cache is True. Otherwise, we can set it to None.
        new_kv = (k, v) if use_cache else None

        #3. Computing Attention scores
        d_k= (self.d_model ** 0.5)
        attn_scores = torch.bmm(q, k.transpose(1, 2))/d_k # (B, T, T_k)

        #4. Applying Causal mask (upper triangular part should be masked) so tokens cannot attend to future positions
        mask = torch.triu(torch.ones((T, k.size(1)), device=x.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))

        #5. Compute attention weights by taking softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        #6. Computing context vectors  
        context = torch.bmm(attn_weights, v) 

        #7.Applying the Output projection
        output = self.W_o(context) # (B, T, C)

        return output, new_kv

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Implement feed-forward network
        # Create two linear layers 
        # Your code here

        # Two linear layers with ReLU in between
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO: Implement the forward pass for feed-forward network
        # Two linear layers with ReLU activation in between
        # Your code here
        
        #Applying linear layer 1 followed by activation function which is followed by linear layer 2
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement decoder layer
        # Create attention, feed-forward, layer norms, and dropout
        # Your code here

        #Single-head self-attention, feed-forward, layer norms, dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = SingleHeadAttention(d_model)
        self.feed_forward= FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder layer
        # 1. Apply layer norm before attention
        # 2. Apply attention followed by dropout
        # 3. Apply layer norm before feed-forward
        # 4. Apply feed-forward then residual connection
        # Your code here

        # 1. Appling layer norm before attention
        norm_x = self.norm1(x)
        # 2. Appling attention followed by dropout
        attn_output, new_kv = self.attention(norm_x, past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(attn_output)
        # 3. Appling layer norm before feed-forward
        norm_x = self.norm2(x)
        # 4. Apply Feed-Forward followed by Residual Connections
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)

        return x, new_kv

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # TODO: Implement decoder-only transformer
        # Create token embedding, positional embedding, decoder layers, output projection
        # Your code here

        # Embeddings 
        self.token_embedding = nn.Embedding(vocab_size, d_model)                 
        self.pos_embedding = nn.Embedding(max_seq_len, d_model) 

        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff) for _ in range(num_layers)]) 

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)  
        # Output projection to vocab size                                         
        self.output_projection = nn.Linear(d_model, vocab_size)    


    def forward(self, input_ids, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder-only transformer
        # 1. Get token embeddings
        # 2. Add positional embeddings (handle position offsets for cached generation)
        # 3. Pass through decoder layers
        # 4. Apply final layer norm
        # 5. Project to vocabulary
        # Your code here

        B, T =  input_ids.shape   #Batch and SeqLen

        # 1. Get token embeddings
        token_emb = self.token_embedding(input_ids)   # (B, T, d_model)

        # 2. Add positional embeddings. If we are using past_kv, we might have a past_len
        past_len = 0
        if past_kv is not None:
            # For the first layer, past_kv[0][0] shape is (B,past_length,d_model)
            past_len = past_kv[0][0].size(1)  # how many tokens are 'cached'
        
        positions = torch.arange(past_len, past_len + T, device=input_ids.device)
        positions = torch.clamp(positions, max=self.pos_embedding.num_embeddings-1)
        pos_embeddings = self.pos_embedding(positions)  # (1,T, d_model)
        
        x = token_emb + pos_embeddings  # (B,T, d_model)

        # 3. Pass through decoder layers
        new_past_kv = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv[i] if past_kv  else None
            x, new_kv = layer(x, past_kv=layer_past_kv, use_cache=use_cache)
            if use_cache:
                new_past_kv.append(new_kv)
        
        # 4. Final layer norm
        x = self.norm(x)

        # 5. Project to vocabulary logits
        logits = self.output_projection(x)

        return logits, new_past_kv
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, use_cache=True):
        # TODO: Implement the generation method
        # 1. Start with the input sequence
        # 2. Iteratively generate new tokens
        # 3. Use temperature for sampling
        # 4. Use KV caching for efficiency
        # Your code here

        self.eval()
        # Initialize past_kv cache
        past_kv = None
         # Start with initial input
        generated = input_ids
        for _ in range(max_new_tokens):
            # Forward pass (take only last token if using cache)
            if use_cache and (past_kv is not None):
                 # Pass only the last token
                next_input = generated[:, -1:]
            else:
                
                next_input = generated
            logits, past_kv = self.forward(next_input, past_kv=past_kv, use_cache=use_cache)

            # Take the last token's logits and apply temperature scaling
            next_token_logits = logits[:, -1, :] / temperature
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append next token
            generated= torch.cat([generated, next_token], dim=1)
        
        return generated


# ================= DO NOT MODIFY CODE BELOW THIS LINE =================
# Evaluation harness code - do not modify

def load_test_cases(filepath):
    """ Load test cases from a file. """
    with open(filepath, 'r') as f:
        test_cases = json.load(f)
    
    # Convert lists back to tensors
    for case in test_cases:
        case['input_ids'] = torch.tensor(case['input_ids'])
        case['expected_logits_no_cache'] = torch.tensor(case['expected_logits_no_cache'])
        case['expected_logits_with_cache'] = torch.tensor(case['expected_logits_with_cache'])
        case['expected_logits_sequential'] = torch.tensor(case['expected_logits_sequential'])
    
    return test_cases

def evaluate_model(model, test_cases, atol=3e-3, with_kv=False):
    """Evaluate model against test cases."""
    model.eval()
    results = []
    
    for i, case in enumerate(test_cases):
        input_ids = case['input_ids']
        expected_logits_no_cache = case['expected_logits_no_cache']
        expected_logits_with_cache = case['expected_logits_with_cache']
        expected_logits_sequential = case['expected_logits_sequential']
        
        with torch.no_grad():
            # Test without caching
            logits_no_cache, _ = model(input_ids, use_cache=False)
            no_cache_match = torch.allclose(logits_no_cache, expected_logits_no_cache, atol=atol)
            
            if with_kv:
                # Test with caching (full sequence)
                logits_with_cache, _ = model(input_ids, use_cache=True)
                with_cache_match = torch.allclose(logits_with_cache, expected_logits_with_cache, atol=atol)

                cache_nocache_match = torch.allclose(logits_no_cache, logits_with_cache, atol=atol)
            
        
        result = {
            'test_case': i + 1,
            'no_cache_match': no_cache_match,
            'with_cache_match': with_cache_match if with_kv else None,
            'cache_nocache_match': cache_nocache_match if with_kv else None,
            'all_match': no_cache_match and (with_cache_match and cache_nocache_match if with_kv else no_cache_match)
        }
        
        if not result['all_match']:
            # Calculate error metrics for debugging
            if not no_cache_match:
                result['no_cache_max_error'] = torch.max(torch.abs(logits_no_cache - expected_logits_no_cache)).item()
            if with_kv and not with_cache_match:
                result['with_cache_max_error'] = torch.max(torch.abs(logits_with_cache - expected_logits_with_cache)).item()
            if with_kv and not cache_nocache_match:
                result['cache_nocache_max_error'] = torch.max(torch.abs(logits_no_cache - logits_with_cache)).item()
        
        results.append(result)
    
    # Overall results
    all_passed = all(r['all_match'] for r in results)
    pass_rate = sum(r['all_match'] for r in results) / len(results)
    
    summary = {
        'all_passed': all_passed,
        'pass_rate': pass_rate,
        'num_test_cases': len(test_cases),
        'num_passed': sum(r['all_match'] for r in results),
        'detailed_results': results
    }
    
    return summary

def benchmark_performance(model, input_ids, num_new_tokens=20, use_cache=True, num_runs=3):
    """Benchmark model performance."""
    model.eval()
    # Warm-up run
    model.generate(input_ids, num_new_tokens, use_cache=use_cache)
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.generate(input_ids, num_new_tokens, use_cache=use_cache)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='Transformer Evaluation Harness')
    parser.add_argument('--mode', type=str, default='run', choices=['generate', 'evaluate', 'kv_evaluate', 'benchmark', 'run'], 
                        help='Mode to run in')
    parser.add_argument('--weights', type=str, default='reference_weights.pt', 
                        help='Path to weights file')
    parser.add_argument('--model_state_dict', type=str, default='model_state_dict.pt', 
                        help='Path to model state dictionary file')
    parser.add_argument('--test_cases', type=str, default='test_cases.json', 
                        help='Path to test cases file')
    parser.add_argument('--vocab_size', type=int, default=1000, 
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=50, 
                        help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=100, 
                        help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of decoder layers')
    parser.add_argument('--max_seq_len', type=int, default=128, 
                        help='Maximum sequence length')
    args = parser.parse_args()
    
    if args.mode == 'generate':
        #Generate evaluation harness -- not accessible to students
        generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
    
    elif args.mode == 'evaluate':
        #Evaluate a model
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        #Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=False)
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        # Print result stats - each test case with pass/fail info
        #print(f"  Detailed results: {results['detailed_results']}")
        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")
    elif args.mode == 'kv_evaluate':
        # Evaluate a model with kv cache against no_kv_cache
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return
        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=True)
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        #detailed results
        #print(f"  Detailed results: {results['detailed_results']}")
        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")
                    if not result.get('with_cache_match', True):
                        print(f"    With cache: Failed (max error: {result.get('with_cache_max_error', 'N/A')})")
    elif args.mode == 'benchmark':
        # Benchmark model performance
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        # Load model state dict
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
            
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return


        # Create sample input
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        # Benchmark with and without caching
        print("Benchmarking...")
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_without_cache / time_with_cache:.2f}x")

        # Adding the code here to Vary the sequence length and benchmark
        sequence_lengths = [10, 50, 100, 500, 1000]
        times_no_cache = []
        times_with_cache = []
        runs = 5
        # Number of new tokens to generate
        num_new_tokens = 40

        print("Benchmarking at different sequence lengths...")
        for seq_len in sequence_lengths:
            # Create random input of shape (1, seq_len)
            input_ids = torch.randint(0, args.vocab_size, (1, seq_len))

            # Measure time without cache
            time_nc = benchmark_performance(
                model, input_ids, num_new_tokens=num_new_tokens, 
                use_cache=False, num_runs=runs
            )

            # Measure time with cache
            time_wc = benchmark_performance(
                model, input_ids, num_new_tokens=num_new_tokens, 
                use_cache=True, num_runs=runs
            )

            times_no_cache.append(time_nc)
            times_with_cache.append(time_wc)

            print(f"SeqLen={seq_len}  NoCache={time_nc:.4f}s  WithCache={time_wc:.4f}s")

        # 4. Plot the results
        plt.figure(figsize=(7,5))
        plt.plot(sequence_lengths, times_no_cache, marker='o', label='No KV Cache')
        plt.plot(sequence_lengths, times_with_cache, marker='o', label='With KV Cache')
        plt.title("Latency vs. Sequence Length (Generating 20 tokens)")
        plt.xlabel("Sequence Length")
        plt.ylabel("Latency (seconds)")
        plt.legend()
        plt.grid(True)
        plt.savefig("kv_cache_latency.png")
        plt.show()
       
    elif args.mode == 'run':
        # Just a debugging mode

        # Default mode: generate harness if files don't exist, then evaluate and benchmark
        if not os.path.exists('model_state_dict.pt') or not os.path.exists('test_cases.json'):
            generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Load model state dict
        state_dict = torch.load('model_state_dict.pt')

        # Print specific weights from state dict
        #print("From state dict - layer 0 Wq weight:")
        #print(state_dict['layers.0.attention.W_q.weight'])
        #print("From state dict - layer 1 Wk weight:")
        #print(state_dict['layers.1.attention.W_k.weight'])

        try:
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model state dictionary from model_state_dict.pt")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
 
        print("Weights loaded from {}".format(args.model_state_dict))

        # print the structure of state_dict
        #print("State dict structure:")
        #print(state_dict.keys())

        # Verify they're the same
        print("Weights match for layer 0 Wq:", 
        torch.allclose(state_dict['layers.0.attention.W_q.weight'], model.layers[0].attention.W_q.weight))
        print("Weights match for layer 1 Wk:", 
        torch.allclose(state_dict['layers.1.attention.W_k.weight'], model.layers[1].attention.W_k.weight))
        
        # Load test cases
        test_cases = load_test_cases('test_cases.json')
        print(f"Test cases loaded from test_cases.json")

        # Print the first test case
        print("First test case:")
        print(test_cases[0].keys())
        # print the tensor shape for each key
        for key in test_cases[0].keys():
            print(key, test_cases[0][key].shape)

        # Evaluate model
        print("\nEvaluating model...")
        results = evaluate_model(model, test_cases)
        
        # Print evaluation results
        print(f"Evaluation Results:")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        
        # Benchmark
        print("\nBenchmarking performance...")
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Performance Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_with_cache > 0 and time_without_cache / time_with_cache:.2f}x")

if __name__ == "__main__":
    main()