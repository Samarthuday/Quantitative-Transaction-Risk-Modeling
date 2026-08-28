#!/usr/bin/env python3
"""
Test script to verify real-time ingestion system
"""

import requests
import time
import json

def test_ingestion():
    print("Testing Real-Time Ingestion System...")
    
    # Test 1: Check API health
    try:
        health = requests.get('http://localhost:5000/api/health').json()
        print(f"✅ API Status: {health['status']}")
        print(f"✅ Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")
        return
    
    # Test 2: Check current stats
    try:
        stats = requests.get('http://localhost:5000/api/monitoring/stats').json()
        print(f"✅ Current Transactions: {stats['total_transactions']}")
    except Exception as e:
        print(f"❌ Stats Check Failed: {e}")
        return
    
    # Test 3: Send test transaction
    try:
        test_tx = {
            'transaction_id': f'test_{int(time.time())}',
            'amount': 25000,
            'sender_id': 'test_user',
            'receiver_id': 'test_recipient',
            'transaction_type': 'payment'
        }
        
        response = requests.post('http://localhost:5000/api/process_transaction', json=test_tx)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Test Transaction Processed: Risk Probability {result['risk_probability']:.3f}")
        else:
            print(f"❌ Test Transaction Failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Test Transaction Error: {e}")
        return
    
    # Test 4: Check updated stats
    try:
        time.sleep(2)
        new_stats = requests.get('http://localhost:5000/api/monitoring/stats').json()
        print(f"✅ Updated Transactions: {new_stats['total_transactions']}")
        
        if new_stats['total_transactions'] > stats['total_transactions']:
            print("✅ Transaction was recorded successfully!")
        else:
            print("❌ Transaction was not recorded!")
            
    except Exception as e:
        print(f"❌ Updated Stats Check Failed: {e}")

if __name__ == "__main__":
    test_ingestion()
