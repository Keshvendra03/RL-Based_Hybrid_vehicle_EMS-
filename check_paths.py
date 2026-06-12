"""
check_paths.py  -- run this from PyCharm to debug path issues
Just right-click this file in PyCharm and click 'Run check_paths'
"""
from pathlib import Path

print("=" * 60)
print("PATH DEBUGGER")
print("=" * 60)

this_file = Path(__file__).resolve()
print(f"\nThis script is at:\n  {this_file}")
print(f"\nSearching for project root (folder with data/ and src/)...")

candidate = this_file.parent
for i in range(8):
    has_data = (candidate / "data").exists()
    has_src  = (candidate / "src").exists()
    print(f"  Level {i}: {candidate}  |  data={has_data}  src={has_src}")
    if has_data and has_src:
        print(f"\n  >> Project root found: {candidate}")
        data_dir = candidate / "data" / "drive_cycles"
        print(f"  >> CSV folder:         {data_dir}")
        print(f"  >> nedc.csv exists:    {(data_dir / 'nedc.csv').exists()}")
        print(f"  >> ftp75.csv exists:   {(data_dir / 'ftp75.csv').exists()}")
        break
    candidate = candidate.parent
else:
    print("\n  >> ERROR: Could not find project root!")

print("\n" + "=" * 60)
