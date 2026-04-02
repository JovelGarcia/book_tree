import {
  Component, OnDestroy, AfterViewInit,
  ElementRef, ViewChild, signal,
  PLATFORM_ID, inject,
} from '@angular/core';
import { isPlatformBrowser }      from '@angular/common';
import { CommonModule }           from '@angular/common';
import { HttpClient }             from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import * as d3                    from 'd3';

// ─── API shapes ─────────────────────────────────────────────────

export interface Relationship {
  id:                number;
  source:            number;
  target:            number;
  source_name:       string;
  target_name:       string;
  relationship_type: string;
  description:       string;
}

export interface MediaDetail {
  id:          number;
  title:       string;
  fandom_url?: string;
}

// ─── Graph primitives ───────────────────────────────────────────

interface GNode extends d3.SimulationNodeDatum {
  id:   number;
  name: string;
  deg:  number;
}

interface GLink extends d3.SimulationLinkDatum<GNode> {
  id:    number;
  type:  string;
  desc:  string;
  sName: string;
  tName: string;
}

// ─── Edge palette ───────────────────────────────────────────────

const PALETTE: Record<string, string> = {
  family:       '#f59e0b',   // amber
  romantic:     '#ec4899',   // pink
  ally:         '#22c55e',   // green
  enemy:        '#ef4444',   // red
  mentor:       '#8b5cf6',   // violet
  rival:        '#ff3d6b',   // hot coral-red (distinct from amber family)
  friend:       '#06b6d4',   // cyan
  political:    '#6366f1',   // indigo
  professional: '#14b8a6',   // teal
  acquaintance: '#e879f9',   // fuchsia (distinct from all above)
  subordinate:  '#facc15',   // yellow (distinct from amber family)
};
const FALLBACK_CLR = '#64748b';
const clr = (t: string) => PALETTE[t?.toLowerCase()] ?? FALLBACK_CLR;

// ─── Component ──────────────────────────────────────────────────

@Component({
  selector:    'app-media-graph',
  standalone:  true,
  imports:     [CommonModule],
  templateUrl: './media-graph.html',
  styleUrl:    './media-graph.css',
})
export class MediaGraph implements AfterViewInit, OnDestroy {

  @ViewChild('container', { static: true }) private boxRef!: ElementRef<HTMLDivElement>;
  @ViewChild('tip',       { static: true }) private tipRef!: ElementRef<HTMLDivElement>;

  loading    = signal(true);
  error      = signal<string | null>(null);
  legend     = signal<{ type: string; color: string }[]>([]);
  mediaTitle = signal<string>('');
  fandomUrl  = signal<string | null>(null);

  private sim:  d3.Simulation<GNode, GLink> | null = null;
  private svg:  d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null;
  private zoom: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null;
  private ro:   ResizeObserver | null = null;
  private roTimer: ReturnType<typeof setTimeout> | null = null;
  private initialResizeFired = false;

  private isBrowser = isPlatformBrowser(inject(PLATFORM_ID));

  constructor(
    private http:   HttpClient,
    private route:  ActivatedRoute,
    private router: Router,
  ) {}

  ngAfterViewInit() {
    if (!this.isBrowser) return;

    const id = +this.route.snapshot.paramMap.get('id')!;

    // non-fatal — header just won't show title if this fails
    this.http.get<MediaDetail>(`/api/media/${id}/`).subscribe({
      next:  m  => { this.mediaTitle.set(m.title); this.fandomUrl.set(m.fandom_url ?? null); },
      error: () => {},
    });

    this.http.get<Relationship[]>(`/api/media/${id}/relationships/`).subscribe({
      next: rels => {
        this.loading.set(false);
        if (!rels.length) { this.error.set('No relationships found.'); return; }
        this.build(rels);
      },
      error: () => {
        this.loading.set(false);
        this.error.set('Failed to load relationship data.');
      },
    });
  }

  ngOnDestroy() {
    this.sim?.stop();
    this.ro?.disconnect();
  }

  goBack()   { this.router.navigate(['/']); }
  fitGraph() { this.zoomToFit(); }

  // ── graph model ──────────────────────────────────────────────

  private build(rels: Relationship[]) {
    const nm = new Map<number, GNode>();
    const ts = new Set<string>();

    for (const r of rels) {
      if (!nm.has(r.source)) nm.set(r.source, { id: r.source, name: r.source_name, deg: 0 });
      if (!nm.has(r.target)) nm.set(r.target, { id: r.target, name: r.target_name, deg: 0 });
      nm.get(r.source)!.deg++;
      nm.get(r.target)!.deg++;
      ts.add(r.relationship_type);
    }

    const nodes = [...nm.values()];
    const links: GLink[] = rels
      .filter(r => r.source !== r.target)
      .map(r => ({
        id:     r.id,
        source: r.source as any,
        target: r.target as any,
        type:   r.relationship_type,
        desc:   r.description,
        sName:  r.source_name,
        tName:  r.target_name,
      }));

    this.legend.set([...ts].sort().map(t => ({ type: t, color: clr(t) })));
    this.render(nodes, links);
  }

  // ── d3 render ────────────────────────────────────────────────

  private render(nodes: GNode[], links: GLink[]) {
    const el  = this.boxRef.nativeElement;
    const tip = this.tipRef.nativeElement;
    const W   = el.clientWidth  || 960;
    const H   = el.clientHeight || 600;
    const cx0 = W / 2;
    const cy0 = H / 2;

    // ── SVG: plain (0,0) origin ──────────────────────────────────
    this.svg = d3.select(el)
      .append('svg')
      .attr('width',   '100%')
      .attr('height',  '100%')
      .attr('viewBox', `0 0 ${W} ${H}`);          // ← KEY FIX

    const g = this.svg.append('g');

    this.zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.05, 6])
      .on('zoom', ({ transform }) => g.attr('transform', transform));

    this.svg.call(this.zoom);

    // seed nodes near canvas center so sim doesn't start at (0,0)
    for (const n of nodes) {
      n.x = cx0 + (Math.random() - 0.5) * 80;
      n.y = cy0 + (Math.random() - 0.5) * 80;
    }

    /* ─── edges ─── */
    const linkSel = g.append('g')
      .selectAll<SVGLineElement, GLink>('line')
      .data(links)
      .join('line')
      .attr('stroke',         d => clr(d.type))
      .attr('stroke-opacity', 0.45)
      .attr('stroke-width',   1.5);

    linkSel
      .on('pointerenter', (ev: PointerEvent, d) => {
        d3.select(ev.currentTarget as Element)
          .attr('stroke-opacity', 1).attr('stroke-width', 3);
        this.showTip(tip, ev, `${d.sName} → ${d.tName}`, d.desc, d.type);
      })
      .on('pointermove',  (ev: PointerEvent) => this.moveTip(tip, ev))
      .on('pointerleave', (ev: PointerEvent) => {
        d3.select(ev.currentTarget as Element)
          .attr('stroke-opacity', 0.45).attr('stroke-width', 1.5);
        this.hideTip(tip);
      });

    /* ─── nodes ─── */
    const maxDeg = d3.max(nodes, n => n.deg) ?? 1;
    const rFn    = (d: GNode) => 5 + 12 * d.deg / maxDeg;

    const nodeSel = g.append('g')
      .selectAll<SVGGElement, GNode>('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'grab')
      .call(this.makeDrag() as any);

    nodeSel.append('circle')
      .attr('r',            rFn)
      .attr('fill',         '#9aa3b2')
      .attr('stroke',       '#4b5263')
      .attr('stroke-width', 1.5);

    nodeSel.append('text')
      .text(d => d.name)
      .attr('text-anchor', 'middle')
      .attr('dx', 0)
      .attr('dy', d => rFn(d) + 13)
      .attr('font-size',     '11px')
      .attr('fill',          '#cbd5e1')
      .attr('pointer-events','none');

    // ── hover: fade non-connected nodes/edges ────────────────
    const connectedIds = (d: GNode): Set<number> => {
      const ids = new Set<number>();
      links.forEach(l => {
        const s = (l.source as GNode).id;
        const t = (l.target as GNode).id;
        if (s === d.id) ids.add(t);
        if (t === d.id) ids.add(s);
      });
      return ids;
    };

    nodeSel
      .on('pointerenter', (ev: PointerEvent, d) => {
        const neighbors = connectedIds(d);
        // fade unrelated nodes
        nodeSel.attr('opacity', n => (n.id === d.id || neighbors.has(n.id)) ? 1 : 0.08);
        // fade unrelated edges
        linkSel.attr('stroke-opacity', l => {
          const s = (l.source as GNode).id;
          const t = (l.target as GNode).id;
          return (s === d.id || t === d.id) ? 0.9 : 0.04;
        }).attr('stroke-width', l => {
          const s = (l.source as GNode).id;
          const t = (l.target as GNode).id;
          return (s === d.id || t === d.id) ? 2.5 : 1.5;
        });
        this.showTip(tip, ev, d.name, `${d.deg} connection${d.deg !== 1 ? 's' : ''}`, '');
      })
      .on('pointermove',  (ev: PointerEvent) => this.moveTip(tip, ev))
      .on('pointerleave', () => {
        nodeSel.attr('opacity', 1);
        linkSel.attr('stroke-opacity', 0.45).attr('stroke-width', 1.5);
        this.hideTip(tip);
      });

    /* ─── force sim ─── forces target canvas center (cx0, cy0) ── */
    this.sim = d3.forceSimulation<GNode, GLink>(nodes)
      .force('x',       d3.forceX<GNode>(cx0).strength(0.1))   // ← KEY FIX
      .force('y',       d3.forceY<GNode>(cy0).strength(0.1))   // ← KEY FIX
      .force('link',    d3.forceLink<GNode, GLink>(links)
                            .id(d => d.id)
                            .distance(250))
      .force('charge',  d3.forceManyBody<GNode>()
                            .strength(-1000)
                            .distanceMin(30)
                            .theta(0.9))
      .force('collide', d3.forceCollide<GNode>(60).strength(0.5))
      .on('tick', () => {
        linkSel
          .attr('x1', d => (d.source as GNode).x!)
          .attr('y1', d => (d.source as GNode).y!)
          .attr('x2', d => (d.target as GNode).x!)
          .attr('y2', d => (d.target as GNode).y!);
        nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
      })
      .on('end', () => this.zoomToFit(600));

    /* ─── responsive viewBox ─── */
    this.initialResizeFired = false;
    this.ro = new ResizeObserver(() => {
      // Skip the very first fire — it's just the initial layout measurement
      // and would reset the viewBox before zoomToFit runs.
      if (!this.initialResizeFired) {
        this.initialResizeFired = true;
        return;
      }

      // Debounce so rapid resize events don't thrash
      if (this.roTimer) clearTimeout(this.roTimer);
      this.roTimer = setTimeout(() => {
        const w = el.clientWidth, h = el.clientHeight;
        if (w && h) {
          this.svg!.attr('viewBox', `0 0 ${w} ${h}`);
          this.zoomToFit(300);   // re-fit after resize
        }
      }, 150);
    });
    this.ro.observe(el);
  }

  // ── zoom to fit ──────────────────────────────────────────────

  private zoomToFit(durationMs = 750) {
    if (!this.sim || !this.svg || !this.zoom) return;
    const nodes = this.sim.nodes();
    if (!nodes.length) return;

    const el  = this.boxRef.nativeElement;
    const W   = el.clientWidth  || 960;
    const H   = el.clientHeight || 600;
    const pad = 60;

    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of nodes) {
      if (n.x! < x0) x0 = n.x!;
      if (n.y! < y0) y0 = n.y!;
      if (n.x! > x1) x1 = n.x!;
      if (n.y! > y1) y1 = n.y!;
    }

    const bw = (x1 - x0) || 1;
    const bh = (y1 - y0) || 1;
    const k  = Math.min((W - 2 * pad) / bw, (H - 2 * pad) / bh, 2);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;

    // translate bounding-box center → viewport center          ← KEY FIX
    const tx = W / 2 - k * cx;
    const ty = H / 2 - k * cy;

    this.svg.transition().duration(durationMs).call(
      this.zoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(k),
    );
  }

  // ── drag ─────────────────────────────────────────────────────

  private makeDrag() {
    const sim = () => this.sim!;
    return d3.drag<SVGGElement, GNode>()
      .on('start', (ev, d) => {
        if (!ev.active) sim().alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag',  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end',   (ev, d) => {
        if (!ev.active) sim().alphaTarget(0);
        d.fx = null; d.fy = null;
      });
  }

  // ── tooltip ───────────────────────────────────────────────────

  private esc(s: string): string {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  private showTip(el: HTMLElement, ev: PointerEvent, title: string, body: string, type: string) {
    const badge = type
      ? `<span class="tip-badge" style="background:${clr(type)}">${this.esc(type)}</span>`
      : '';
    el.innerHTML = `${badge}<strong>${this.esc(title)}</strong><p>${this.esc(body)}</p>`;
    el.style.opacity = '1';
    this.moveTip(el, ev);
  }

  private moveTip(el: HTMLElement, ev: PointerEvent) {
    el.style.left = `${ev.clientX + 14}px`;
    el.style.top  = `${ev.clientY + 14}px`;
  }

  private hideTip(el: HTMLElement) { el.style.opacity = '0'; }
}
