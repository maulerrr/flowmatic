# Flowmatic Redesign - Developer Implementation Notes

## Quick Start

### Running the Application
```bash
cd frontend
bun install
bun run dev
```

The application will be available at `http://localhost:5173`

---

## Color Variables

All colors are defined as CSS custom properties in `src/styles.css`:

```css
--primary: #34d0c3
--secondary: #7c8cff
--success: #36c28a
--warning: #f2c14f
--destructive: #f05d5e
--background: #0d141f
--foreground: #eaf2ff
--card: #131d2b
--border: #1f2a3a
```

Usage in Tailwind:
```vue
<!-- Color classes -->
class="bg-primary text-foreground border-border"

<!-- Opacity modifiers -->
class="bg-primary/20 text-foreground/60"
```

---

## Spacing System

Based on 4px unit:
- `p-2` = 8px
- `p-4` = 16px
- `p-6` = 24px
- `p-8` = 32px
- `gap-3` = 12px
- `gap-4` = 16px
- `gap-6` = 24px

---

## Animation Classes

### Transitions
```vue
<!-- Smooth color transitions -->
<button class="transition-colors duration-200">

<!-- All property transitions -->
<div class="transition-all duration-300">

<!-- Scale on hover -->
<div class="hover:scale-110 transition-transform">

<!-- Lift effect -->
<button class="hover:-translate-y-0.5 transition-transform">
```

### Animation Components
```vue
<!-- Sliding fade animation -->
<transition name="slide-fade">
  <div v-if="visible">Content</div>
</transition>

<!-- Simple fade -->
<transition name="fade">
  <div v-if="visible">Content</div>
</transition>

<!-- Scale animation -->
<transition name="scale">
  <div v-if="visible">Content</div>
</transition>
```

### Pulse Effect
```vue
<!-- Animated pulse -->
<div class="animate-pulse">

<!-- Glow pulse -->
<div class="animate-glow-pulse">
```

---

## Component Usage Examples

### StatCard Component
```vue
<script setup lang="ts">
import StatCard from '@/core/components/stat-card.vue'
import { FileText } from 'lucide-vue-next'
</script>

<template>
  <StatCard
    label="Total Uploads"
    :value="24"
    :icon="FileText"
    color="primary"
    description="this month"
    :trend="{ value: 20, positive: true }"
  />
</template>
```

**Props:**
- `label` (string) - Label text
- `value` (string | number) - Display value
- `icon` (Component) - Icon component
- `color` ('primary' | 'success' | 'warning' | 'destructive')
- `description` (string, optional) - Subtitle
- `trend` (object, optional) - Trend data with `value` and `positive`

---

### PipelineRunCard Component
```vue
<script setup lang="ts">
import PipelineRunCard from '@/core/components/pipeline-run-card.vue'
import type { PipelineRun } from '@/api/client'

const run: PipelineRun = { /* ... */ }

function handleDelete() {
  // Delete logic
}
</script>

<template>
  <PipelineRunCard
    :run="run"
    :menuOpen="false"
    @menu-toggle="toggleMenu"
    @preview="showDataPreview(run)"
    @export="openExportModal(run)"
    @delete="handleDelete"
  />
</template>
```

**Props:**
- `run` (PipelineRun) - Run object
- `menuOpen` (boolean) - Menu state

**Events:**
- `menu-toggle` - Toggle dropdown menu
- `preview` - Preview data
- `export` - Export data
- `delete` - Delete run

---

## Form Elements

### Input Styling
```vue
<!-- Standard input -->
<input
  type="text"
  class="w-full px-4 py-2 rounded-lg border border-border bg-input focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/30 transition-all"
/>

<!-- Disabled input -->
<input
  type="email"
  disabled
  class="w-full px-4 py-2 rounded-lg border border-border bg-border/20 text-foreground/70 disabled:cursor-not-allowed"
/>

<!-- Textarea -->
<textarea
  class="w-full px-4 py-2 rounded-lg border border-border bg-input focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/30 transition-all resize-none"
/>
```

### Button Styling
```vue
<!-- Primary button -->
<button class="px-6 py-2 rounded-lg bg-gradient-to-r from-primary to-secondary text-foreground font-medium hover:shadow-[var(--glow)] transition">
  Action
</button>

<!-- Secondary button -->
<button class="px-6 py-2 rounded-lg border border-border hover:border-primary/40 text-foreground font-medium transition">
  Action
</button>

<!-- Danger button -->
<button class="px-6 py-2 rounded-lg bg-destructive/10 border border-destructive/30 hover:bg-destructive/20 text-destructive font-medium transition">
  Delete
</button>

<!-- Disabled button -->
<button disabled class="px-6 py-2 rounded-lg bg-primary/50 text-foreground font-medium cursor-not-allowed opacity-50">
  Loading...
</button>
```

---

## Status Indicators

```vue
<!-- Success status -->
<span class="px-3 py-1 rounded-lg bg-success/20 text-success text-xs font-semibold">
  Completed
</span>

<!-- Warning status -->
<span class="px-3 py-1 rounded-lg bg-warning/20 text-warning text-xs font-semibold">
  Queued
</span>

<!-- Error status -->
<span class="px-3 py-1 rounded-lg bg-destructive/20 text-destructive text-xs font-semibold">
  Failed
</span>

<!-- Info status -->
<span class="px-3 py-1 rounded-lg bg-primary/20 text-primary text-xs font-semibold">
  Processing
</span>
```

---

## Cards & Containers

### Standard Card
```vue
<div class="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6 hover:border-primary/40 transition-all">
  <!-- Content -->
</div>
```

### Card with Header
```vue
<div class="rounded-xl border border-border bg-card/40 backdrop-blur-sm overflow-hidden">
  <div class="px-6 py-4 border-b border-border/30 bg-card/50">
    <h3 class="text-lg font-bold text-foreground">Title</h3>
  </div>
  <div class="p-6">
    <!-- Content -->
  </div>
</div>
```

### Glass Effect Container
```vue
<div class="rounded-xl border border-border/50 bg-card/40 backdrop-blur-xl shadow-lg">
  <!-- Content -->
</div>
```

---

## Layout Patterns

### Two-Column Layout
```vue
<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
  <div>Column 1</div>
  <div>Column 2</div>
</div>
```

### Three-Column Layout
```vue
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  <div>Column 1</div>
  <div>Column 2</div>
  <div>Column 3</div>
</div>
```

### Header with Subtitle
```vue
<div class="px-6 py-8 border-b border-border/50">
  <div class="max-w-7xl mx-auto">
    <h1 class="text-3xl md:text-4xl font-bold text-foreground mb-2">Title</h1>
    <p class="text-foreground/60">Subtitle or description</p>
  </div>
</div>
```

---

## Icons

Using Lucide Vue Next:

```vue
<script setup lang="ts">
import { FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-vue-next'
</script>

<template>
  <!-- Icon sizing -->
  <FileText class="w-5 h-5" />
  <FileText class="w-6 h-6" />
  <FileText class="w-8 h-8" />
  
  <!-- Icon colors -->
  <FileText class="text-primary" />
  <FileText class="text-success" />
  <FileText class="text-destructive" />
  <FileText class="text-foreground/60" />
  
  <!-- Icon animations -->
  <Loader2 class="animate-spin" />
</template>
```

---

## Responsive Design

### Breakpoints
- `sm:` - 640px and up
- `md:` - 768px and up
- `lg:` - 1024px and up
- `xl:` - 1280px and up

### Mobile-First Example
```vue
<!-- Stack on mobile, side-by-side on tablet+ -->
<div class="flex flex-col md:flex-row gap-4">
  <div class="flex-1">Item 1</div>
  <div class="flex-1">Item 2</div>
</div>

<!-- Hidden on mobile, visible on desktop -->
<button class="hidden lg:inline-flex">Desktop Only</button>

<!-- Different padding at different sizes -->
<div class="px-4 md:px-6 lg:px-8 py-4 md:py-6 lg:py-8">
  Content
</div>
```

---

## Accessibility Tips

### Focus States
```vue
<button class="focus:outline-none focus:ring-2 focus:ring-primary/50 rounded-lg">
  Click me
</button>
```

### Semantic HTML
```vue
<!-- Use correct heading hierarchy -->
<h1>Page Title</h1>
<h2>Section</h2>
<h3>Subsection</h3>

<!-- Use semantic tags -->
<nav>Navigation</nav>
<main>Main content</main>
<footer>Footer</footer>

<!-- Always use labels for forms -->
<label for="email">Email</label>
<input id="email" type="email" />
```

### ARIA Labels
```vue
<!-- For icon-only buttons -->
<button :aria-label="'Delete run ' + run.id">
  <Trash2 class="w-5 h-5" />
</button>

<!-- For loading states -->
<div :aria-live="'polite'" :aria-busy="loading">
  <Loader2 v-if="loading" class="animate-spin" />
  <span v-else>Loaded</span>
</div>
```

---

## Common Patterns

### Loading State
```vue
<script setup lang="ts">
const loading = ref(false)
</script>

<template>
  <div v-if="loading" class="rounded-xl border border-border bg-card/40 p-8 text-center">
    <Loader2 class="w-8 h-8 text-primary animate-spin mx-auto mb-4" />
    <p class="text-foreground/60">Loading...</p>
  </div>
</template>
```

### Empty State
```vue
<template>
  <div class="rounded-xl border border-border bg-card/40 p-12 text-center">
    <FileText class="w-16 h-16 text-foreground/40 mx-auto mb-4" />
    <h3 class="text-lg font-bold text-foreground mb-2">No Data</h3>
    <p class="text-foreground/60 mb-6">Start by uploading a file</p>
    <router-link to="/upload" class="px-4 py-2 rounded-lg bg-primary text-foreground font-medium">
      Upload Now
    </router-link>
  </div>
</template>
```

### Error Message
```vue
<template>
  <div class="flex items-start gap-3 bg-destructive/10 border border-destructive/30 rounded-lg p-4">
    <AlertCircle class="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
    <p class="text-destructive text-sm">{{ errorMessage }}</p>
  </div>
</template>
```

### Success Message
```vue
<template>
  <div class="flex items-start gap-3 bg-success/10 border border-success/30 rounded-lg p-4">
    <CheckCircle class="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
    <p class="text-success text-sm">{{ successMessage }}</p>
  </div>
</template>
```

---

## Debugging Tips

### Tailwind Classes Not Applied?
1. Make sure the file is in `src/` directory
2. Check that class names are complete (no string interpolation)
3. Run `bun run dev` to rebuild CSS

### Animation Lag?
1. Use `will-change` for animated elements
2. Keep animations under 300ms
3. Use `transform` and `opacity` for best performance

### Mobile Layout Issues?
1. Always test with DevTools device toolbar
2. Check breakpoint assumptions
3. Ensure touch targets are 44px+

### Colors Not Right?
1. Check CSS custom property spelling
2. Verify opacity values
3. Test in dark mode

---

## Performance Checklist

- [ ] All animations use GPU acceleration
- [ ] No unnecessary re-renders
- [ ] Images optimized (use WebP)
- [ ] CSS is purged (no unused classes)
- [ ] Bundle size reasonable
- [ ] Lighthouse score > 90

---

## Testing Checklist

- [ ] Test on mobile (320px+)
- [ ] Test on tablet (768px+)
- [ ] Test on desktop (1280px+)
- [ ] Test keyboard navigation
- [ ] Test with screen reader
- [ ] Test dark mode (if applicable)
- [ ] Test all form inputs
- [ ] Test error states
- [ ] Test loading states
- [ ] Test animations (smooth?)

---

## Common Issues & Solutions

### Issue: Animations feel jerky
**Solution:** Add `transform: translateZ(0)` or use `will-change`

### Issue: Text too small on mobile
**Solution:** Use responsive text sizes: `text-sm md:text-base`

### Issue: Buttons overlapping on mobile
**Solution:** Use `flex-col md:flex-row` and adjust padding

### Issue: Colors look wrong
**Solution:** Check for opacity overrides or inherited colors

### Issue: Border colors too light
**Solution:** Use `/50` opacity instead of `/30` for borders

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/redesign-login

# Make changes
git add src/pages/login-page.vue

# Commit with clear message
git commit -m "feat: redesign login page with dark theme"

# Push to remote
git push origin feature/redesign-login

# Create pull request
```

---

## Resources

- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Vue 3 Docs](https://vuejs.org/guide/introduction.html)
- [Lucide Icons](https://lucide.dev/)
- [WCAG Accessibility](https://www.w3.org/WAI/WCAG21/quickref/)
- [CSS Animations](https://developer.mozilla.org/en-US/docs/Web/CSS/animation)

---

**Last Updated:** January 14, 2026  
**Version:** 1.0
