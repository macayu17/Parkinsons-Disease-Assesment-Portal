# How to Create a Pull Request via Fork

## Step 1: Fork the Repository

1. Go to: https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal
2. Click the **"Fork"** button in the top-right corner
3. This creates a copy under your account: `https://github.com/YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal`

## Step 2: Update Your Local Repository

```powershell
# Remove the old remote
git remote remove origin

# Add your fork as the new origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal.git

# Add the original repo as upstream
git remote add upstream https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal.git

# Verify remotes
git remote -v
```

## Step 3: Push to Your Fork

```powershell
# Push your branch to YOUR fork
git push -u origin feature/cuda-training-optimization
```

This will work without authentication issues since it's your own repository!

## Step 4: Create Pull Request

1. Go to YOUR fork: `https://github.com/YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal`
2. You'll see: **"feature/cuda-training-optimization had recent pushes"**
3. Click **"Compare & pull request"**
4. GitHub will automatically set:
   - **Base repository**: `macayu17/Parkinsons-Disease-Assesment-Portal` (base: `main`)
   - **Head repository**: `YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal` (compare: `feature/cuda-training-optimization`)
5. Copy the content from `PULL_REQUEST_TEMPLATE.md` into the PR description
6. Click **"Create pull request"**

## Step 5: Owner Approves

Once you create the pull request:
- **macayu17** will receive a notification
- They can review your changes
- They can approve and merge the pull request
- Your changes will then be in the main repository!

## Quick Commands Summary

```powershell
# After forking on GitHub:
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal.git
git remote add upstream https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal.git
git push -u origin feature/cuda-training-optimization
```

Then create the PR on GitHub's web interface!

## Benefits of This Approach

✅ **No Permission Issues** - You push to your own fork  
✅ **Standard Workflow** - This is how most open-source contributions work  
✅ **Full Control** - You can make more changes before the owner reviews  
✅ **Clear History** - The PR shows exactly what you're proposing  
✅ **Review Process** - The owner can comment and request changes  

## What Happens Next

1. You create the PR from your fork → original repo
2. Owner (macayu17) gets notified
3. They review your code changes
4. They can:
   - ✅ **Approve & Merge** - Your changes go into main branch
   - 💬 **Comment** - Ask questions or request changes
   - ❌ **Close** - Decline the changes (rare if well-documented)
5. Once merged, your CUDA improvements are live! 🎉

---

**TIP**: The PR template I created (`PULL_REQUEST_TEMPLATE.md`) has all the details about your changes, performance improvements, and testing - this makes it easy for the owner to understand and approve!
