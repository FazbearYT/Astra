import sklearn
print(f"scikit-learn версия: {sklearn.__version__}")

from sklearn.linear_model import LogisticRegression
import inspect

# Проверяем параметры
sig = inspect.signature(LogisticRegression.__init__)
params = list(sig.parameters.keys())
print(f"\nДоступные параметры LogisticRegression:")
for param in params:
    if param != 'self':
        print(f"  - {param}")

if 'multi_class' in params:
    print("\n✅ Параметр 'multi_class' доступен")
else:
    print("\n⚠️  Параметр 'multi_class' УДАЛЕН (sklearn >= 1.5)")
    print("   Не используйте этот параметр!")